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

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/*
0x06 Combiner
如果需要做 mean pooling，则需需要做两个操作。
 *   1) forward
 *        sum: calling forward_sum_kernel()
 *        mean: calling foward_sum_kernel() + forward_scale_kernel()
第一个操作是对CSR row offset 做一个AllReduce，这样就相当于是一个全局offset了，就可以拿到每个sample每个slot里的key的总个数。
第二个操作是Forward Scale，就是把embedding的值除以这个"个数"，也就等于做了平均。
    // scale for combiner=mean after reduction
    if (embedding_data_.embedding_params_.combiner == 1) {
   size_t send_count = embedding_data_.embedding_params_.get_batch_size(is_train) *
                           embedding_data_.embedding_params_.slot_num +
                       1;
   functors_.all_reduce(send_count, embedding_data_.get_row_offsets_tensors(is_train),
                        row_offset_allreduce_tensors_, embedding_data_.get_resource_manager());

   // do average
   functors_.forward_scale(
       embedding_data_.embedding_params_.get_batch_size(is_train),
       embedding_data_.embedding_params_.slot_num,
       embedding_data_.embedding_params_.embedding_vec_size, row_offset_allreduce_tensors_,
       embedding_data_.get_output_tensors(is_train), embedding_data_.get_resource_manager());
 }
 6.1 AllReduce
AllReduce 结果如下：
006-014.png
回忆一下 CSR 例子。
    *   40,50,10,20
    *   30,50,10
    *   30,20
    *   10
    * Will be convert to the form of:
    * row offset: 0,4,7,9,10
    * value: 40,50,10,20,30,50,10,30,20,10
row_offset 的数字就是：第一行起始位置是0，第二行起始位置是4，第三行起始位置是7..... 我们假设这是在Node 1之上。
如果Node 2的row_offset为 0,5,7,10,11，说明在这个Node之上，第一行起始位置是0，第二行起始位置是5，第三行起始位置是7.....，对应CSR是：
    *   40,50,10,20,30
    *   30,50
    *   30,20,40
    *   10
    * Will be convert to the form of:
    * row offset: 0,5,7,10,11
    * value: 40,50,10,20,30,50,10,30,20,10
做了AllReduce之后，得到：0,9,14,19,21。这样就知道第一个行总个数是9个，第二行总个是是7+7-9 = 5个。
具体算子如下：
 */
/**
 * collection communication: all_reduce.
 * @param send_count the count of elements will be sent.
 * @param send_tensors the send tensors of multi GPUs.
 * @param recv_tensors the recv tensors of multi GPUs.
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename TypeHashKey>
void SparseEmbeddingFunctors::all_reduce(size_t send_count,
                                         const Tensors2<TypeHashKey> &send_tensors,
                                         Tensors2<TypeHashKey> &recv_tensors,
                                         const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  // need to know the type of Type here
  ncclDataType_t type;
  switch (sizeof(TypeHashKey)) {
    case 4:
      type = ncclUint32;
      break;
    case 8:
      type = ncclUint64;
      break;
    default:
      CK_THROW_(Error_t::WrongInput, "Error: Type not support by now");
  }

  // for multi GPUs, use NCCL to do all_reduce (supporting multi-node GPU servers)
  if (total_gpu_count > 1) {
    CK_NCCL_THROW_(ncclGroupStart());
    for (size_t id = 0; id < local_gpu_count; id++) {
      const auto &local_gpu = resource_manager.get_local_gpu(id);
      // ALLReduce操作
      CK_NCCL_THROW_(ncclAllReduce(send_tensors[id].get_ptr(), recv_tensors[id].get_ptr(),
                                   send_count, type, ncclSum, local_gpu->get_nccl(),
                                   local_gpu->get_stream()));
    }
    CK_NCCL_THROW_(ncclGroupEnd());
  }
  // for single GPU, just do memcpyD2D
  else {  // total_gpu_count == 1
    const auto &local_gpu = resource_manager.get_local_gpu(0);
    CudaDeviceContext context(local_gpu->get_device_id());
    CK_CUDA_THROW_(cudaMemcpyAsync(recv_tensors[0].get_ptr(), send_tensors[0].get_ptr(),
                                   send_count * sizeof(TypeHashKey), cudaMemcpyDeviceToDevice,
                                   local_gpu->get_stream()));
  }

  return;
}

template void SparseEmbeddingFunctors::all_reduce<unsigned int>(
    size_t send_count, const Tensors2<unsigned int> &send_tensors,
    Tensors2<unsigned int> &recv_tensors, const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all_reduce<long long>(
    size_t send_count, const Tensors2<long long> &send_tensors, Tensors2<long long> &recv_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR