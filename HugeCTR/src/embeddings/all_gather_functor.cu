/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <omp.h>

#include <cub/cub.cuh>
#include <numeric>

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

/*
4.2 AllGather
反向传播的第一步是使用 all-gather 操作来在每个 GPU 之上都收集到的所有样本的全部梯度，这样后续可以进行计算并且更新每个 GPU 之上的参数。
4.2.1 原理
首先我们看 AllGather 原理。在执行 AllGather 操作时，K个处理器之中，每个处理器都会将来自每个处理器的N个值聚集成一个维度为K*N的输出。
输出是按rank索引排序的。AllGather操作会受到不同rank或设备映射的影响，因为rank决定了数据布局。
注意：执行ReduceScatter + AllGather，就等同于AllReduce操作。
007-002-A.png
4.2.2 代码
调用代码如下，可以看到其会把梯度从反向传播的输入 embedding_data_.get_output_tensors(true) 拷贝到 embedding_feature_tensors_。
因此，embedding_feature_tensors_ 将会拥有所有的梯度。
    functors_.all_gather(send_count, embedding_data_.get_output_tensors(true),
                         embedding_feature_tensors_, embedding_data_.get_resource_manager());
算子如下
 */
/**
 * collection communication: all_gather.
 * @param send_count the count of elements will be sent.
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename Type>
void SparseEmbeddingFunctors::all_gather(size_t send_count, const Tensors2<Type> &send_tensors,
                                         Tensors2<Type> &recv_tensors,
                                         const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  // need to know the Type
  ncclDataType_t type;
  switch (sizeof(Type)) {
    case 2:
      type = ncclHalf;
      break;
    case 4:
      type = ncclFloat;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
  }

  // for multi GPUs, use NCCL to do All-Gather
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_manager.get_local_gpu(id);
      CK_NCCL_THROW_(ncclAllGather(send_tensors[id].get_ptr(),  // send buff
                                   recv_tensors[id].get_ptr(),  // recv buff
                                   send_count, type, local_gpu->get_nccl(),
                                   local_gpu->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  // for single GPU, just do memcpyD2D
  else {  // total_gpu_count == 1
    const auto &local_gpu = resource_manager.get_local_gpu(0);
    CudaDeviceContext context(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0].get_ptr(), send_tensors[0].get_ptr(),
                                   send_count * sizeof(Type), cudaMemcpyDeviceToDevice,
                                   local_gpu->get_stream()));
  }

  return;
}

template <typename Type>
void SparseEmbeddingFunctors::prepare_for_sparse_all_gather(
    const SparseTensors<Type> &send_tensors, SparseTensorAllGatherConfig<Type> &config,
    const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  if (send_tensors.size() != local_gpu_count) {
    CK_THROW_(Error_t::OutOfBound, "prepare_for_sparse_all_gather send tensors check error");
  }

  if (local_gpu_count == total_gpu_count) {
    size_t total_nnz = 0;
    for (size_t i = 0; i < local_gpu_count; ++i) {
      config.nnzs.get_ptr()[i] = total_nnz;
      config.nnzs_num.get_ptr()[i] = send_tensors[i].nnz();
      total_nnz += send_tensors[i].nnz();
    }
    config.total_nnz = total_nnz;
    return;
  }

#ifdef ENABLE_MPI
  std::vector<size_t> local_nnzs(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; ++id) {
    local_nnzs[id] = send_tensors[id].nnz();
  }
  std::vector<size_t> global_nnzs(total_gpu_count, 0);

  CK_MPI_THROW_(MPI_Allgather(local_nnzs.data(), local_gpu_count * sizeof(size_t), MPI_CHAR,
                              global_nnzs.data(), total_gpu_count * sizeof(size_t), MPI_CHAR,
                              MPI_COMM_WORLD));

  size_t total_nnz = 0;
  for (size_t i = 0; i < total_gpu_count; ++i) {
    config.nnzs.get_ptr()[i] = total_nnz;
    config.nnzs_num.get_ptr()[i] = global_nnzs[i];
    total_nnz += global_nnzs[i];
  }
  config.total_nnz = total_nnz;
#endif
}

namespace sparse_tensor_all_gather_kernel {
template <typename Type>
__global__ void split_rowoffset(const Type *rowoffset, size_t rowoffset_count,
                                Type *rowoffset_split_result) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < rowoffset_count) {
    rowoffset_split_result[tid] = rowoffset[tid + 1] - rowoffset[tid];
  }
}
}  // namespace sparse_tensor_all_gather_kernel

// for sparse tensor, this is an all_gatherv
// prepare SparseTensorAllGatherConfig.host_nnz first
template <typename Type>
void SparseEmbeddingFunctors::all_gather(const SparseTensor<Type> &send_tensor,
                                         SparseTensor<Type> &recv_tensor,
                                         SparseTensorAllGatherConfig<Type> &config, size_t id,
                                         const ResourceManager &resource_manager,
                                         cudaStream_t stream) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  const auto &local_gpu = resource_manager.get_local_gpu(id);
  CudaDeviceContext context(local_gpu->get_device_id());

  ncclDataType_t nccl_type = NcclDataType<Type>::getType();
  size_t send_rowoffset_num = send_tensor.rowoffset_count() - 1;
  {
    constexpr int block_size = 256;
    int grid_size = (send_rowoffset_num - 1) / block_size + 1;
    sparse_tensor_all_gather_kernel::split_rowoffset<<<grid_size, block_size, 0, stream>>>(
        send_tensor.get_rowoffset_ptr(), send_rowoffset_num,
        recv_tensor.get_rowoffset_ptr() + id * send_rowoffset_num);
  }
  {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t recv_id = 0; recv_id < total_gpu_count; ++recv_id) {
      CK_NCCL_THROW_(ncclBroadcast(
          send_tensor.get_value_ptr(), recv_tensor.get_value_ptr() + config.nnzs.get_ptr()[recv_id],
          config.nnzs_num.get_ptr()[recv_id], nccl_type, recv_id, local_gpu->get_nccl(), stream));
    }

    CK_NCCL_THROW_(ncclAllGather(
        recv_tensor.get_rowoffset_ptr() + id * send_rowoffset_num, recv_tensor.get_rowoffset_ptr(),
        send_rowoffset_num, nccl_type, local_gpu->get_nccl(),
        stream));  // send_rowoffset_num may vary between train and evaluate
    CK_NCCL_THROW_(ncclGroupEnd());
  }

  *recv_tensor.get_nnz_ptr() = config.total_nnz;

  return;
}

template void SparseEmbeddingFunctors::all_gather<float>(size_t send_count,
                                                         const Tensors2<float> &send_tensors,
                                                         Tensors2<float> &recv_tensors,
                                                         const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all_gather<__half>(size_t send_count,
                                                          const Tensors2<__half> &send_tensors,
                                                          Tensors2<__half> &recv_tensors,
                                                          const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::prepare_for_sparse_all_gather<unsigned int>(
    const SparseTensors<unsigned int> &send_tensors,
    SparseTensorAllGatherConfig<unsigned int> &config, const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::prepare_for_sparse_all_gather<long long>(
    const SparseTensors<long long> &send_tensors, SparseTensorAllGatherConfig<long long> &config,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all_gather<unsigned int>(
    const SparseTensor<unsigned int> &send_tensor, SparseTensor<unsigned int> &recv_tensor,
    SparseTensorAllGatherConfig<unsigned int> &config, size_t id,
    const ResourceManager &resource_manager, cudaStream_t stream);

template void SparseEmbeddingFunctors::all_gather<long long>(
    const SparseTensor<long long> &send_tensor, SparseTensor<long long> &recv_tensor,
    SparseTensorAllGatherConfig<long long> &config, size_t id,
    const ResourceManager &resource_manager, cudaStream_t stream);

}  // namespace HugeCTR
