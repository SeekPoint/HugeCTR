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

namespace HugeCTR {

#ifdef ENABLE_MPI

template <typename Type>
void SparseEmbeddingFunctors::all2all_forward(size_t batch_size_per_gpu, size_t slot_num,
                                              size_t embedding_vec_size,
                                              const Tensors2<Type> &send_tensors,
                                              Tensors2<Type> &recv_tensors,
                                              const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();

  size_t num_proc = resource_manager.get_num_process();
  if (total_gpu_count != (num_proc * local_gpu_count)) {
    CK_THROW_(Error_t::WrongInput, "Error: the total gpu count doesn't match");
  }

  std::vector<const Type *> src(local_gpu_count);
  std::vector<Type *> dst(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src[id] = send_tensors[id].get_ptr();
    dst[id] = recv_tensors[id].get_ptr();
  }

  std::vector<std::vector<size_t>> send_table(local_gpu_count,
                                              std::vector<size_t>(total_gpu_count));
  std::vector<std::vector<size_t>> recv_table(local_gpu_count,
                                              std::vector<size_t>(total_gpu_count));

  // Fill in sending partition table, ith Topo GPU send to jth global GPU
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t global_id = resource_manager.get_local_gpu(i)->get_global_id();
    size_t slot_num_per_gpu =
        slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
    size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

    for (size_t j = 0; j < total_gpu_count; j++) {
      send_table[i][j] = element_per_send;
    }
  }

  // Fill in receiving partition table, ith Topo GPU receive from jth global GPU
  for (size_t j = 0; j < total_gpu_count; j++) {
    size_t global_id = j;
    size_t slot_num_per_gpu =
        slot_num / total_gpu_count + ((global_id < (slot_num % total_gpu_count)) ? 1 : 0);
    size_t element_per_recv = batch_size_per_gpu * slot_num_per_gpu * embedding_vec_size;

    for (size_t i = 0; i < local_gpu_count; i++) {
      recv_table[i][j] = element_per_recv;
    }
  }

  std::vector<std::vector<const Type *>> src_pos(local_gpu_count,
                                                 std::vector<const Type *>(total_gpu_count));
  std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(total_gpu_count));
  // Calculate the src offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t src_offset = 0;
    for (size_t j = 0; j < total_gpu_count; j++) {
      src_pos[i][j] = src[i] + src_offset;
      src_offset += send_table[i][j];
    }
  }
  // Calculate the dst offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t dst_offset = 0;
    for (size_t j = 0; j < total_gpu_count; j++) {
      dst_pos[i][j] = dst[i] + dst_offset;
      dst_offset += recv_table[i][j];
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all forward src_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << src_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "nccl all2all forward dst_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < total_gpu_count; j++) {
      std::cout << dst_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

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

  // Do the all2all transfer
  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    const auto &local_gpu = resource_manager.get_local_gpu(i);
    for (size_t j = 0; j < total_gpu_count; j++) {
      CK_NCCL_THROW_(ncclSend(src_pos[i][j], send_table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
      CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], recv_table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
    }
  }
  CK_NCCL_THROW_(ncclGroupEnd());

  return;
}

template void SparseEmbeddingFunctors::all2all_forward<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_forward<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#else

/*
4.2 alltoall
因为 forward_per_gpu 函数已经在前文介绍过，所以我们直接来看 alltoall操作。

我们前文介绍过，每个GPU在本地获取到稠密向量之后，会存入 embedding_feature_tensors_。
这是一维数组，在 dist 类型下，长度为 sample_num（batch_size） * slot_num_per_gpu[i] * embedding_vec_size。
在local这里就是：batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size。

所以接下来就要在各个GPU之间彼此发送 embedding_feature_tensors_，然后每个GPU只接受自己应该接受的。

MPI_Alltoall与MPI_AllGahter相比较，区别在于：
      MPI_AllGather：不同进程从某一进程（聚集结果进程）收集到的数据完全相同。
      MPI_Alltoall：不同的进程从某一进程（聚集结果进程）收集到的数据不同。
比如发送的是：
            rank=0, 发送 0 1 2
            rank=1, 发送 3 4 5
            rank=2, 发送 6 7 8
则接受的是：
            rank=0, 接受 0 3 6
            rank=1, 接受 1 4 7
            rank=2, 接受 2 5 8
针对我们的例子，目前如下：

    GPU0发送：1,3,5,7
    GPU1发送：2,4,6,8

    GPU0接受：1,3,2,4
    GPU1接受：5,7,6,8
得到如下，"..." 代表 all2all_tensors_ 长度不止是4个item。
008-003
*/
template <typename Type>
void SparseEmbeddingFunctors::all2all_forward(size_t batch_size_per_gpu,
                                              const std::vector<size_t> &slot_num_per_gpu,
                                              size_t embedding_vec_size,
                                              const Tensors2<Type> &send_tensors,
                                              Tensors2<Type> &recv_tensors,
                                              const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  // Fill in partition table, ith Topo GPU to jth Topo GPU
  std::vector<std::vector<size_t>> table(local_gpu_count, std::vector<size_t>(local_gpu_count));
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t element_per_send = batch_size_per_gpu * slot_num_per_gpu[i] * embedding_vec_size;
    for (size_t j = 0; j < local_gpu_count; j++) {
      table[i][j] = element_per_send;
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all forward table:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << table[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

  std::vector<const Type *> src(local_gpu_count);
  std::vector<Type *> dst(local_gpu_count);
  for (size_t id = 0; id < local_gpu_count; id++) {
    src[id] = send_tensors[id].get_ptr();
    dst[id] = recv_tensors[id].get_ptr();
  }
  std::vector<std::vector<const Type *>> src_pos(local_gpu_count,
                                                 std::vector<const Type *>(local_gpu_count));
  std::vector<std::vector<Type *>> dst_pos(local_gpu_count, std::vector<Type *>(local_gpu_count));

  // 设定源数据的offset
  // Calculate the src offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t src_offset = 0;
    for (size_t j = 0; j < local_gpu_count; j++) {
      src_pos[i][j] = src[i] + src_offset;
      src_offset += table[i][j];
    }
  }

  // 设定目标数据的offset
  // Calculate the dst offset pointer from each GPU to each other
  for (size_t i = 0; i < local_gpu_count; i++) {
    size_t dst_offset = 0;
    for (size_t j = 0; j < local_gpu_count; j++) {
      dst_pos[i][j] = dst[i] + dst_offset;
      dst_offset += table[j][i];
    }
  }

#ifndef NDEBUG
  std::cout << "nccl all2all forward src_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << src_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "nccl all2all forward dst_pos:" << std::endl;
  for (size_t i = 0; i < local_gpu_count; i++) {
    for (size_t j = 0; j < local_gpu_count; j++) {
      std::cout << dst_pos[i][j] << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif

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

  // Do the all2all transfer
  CK_NCCL_THROW_(ncclGroupStart());
  for (size_t i = 0; i < local_gpu_count; i++) {
    const auto &local_gpu = resource_manager.get_local_gpu(i);
    PROFILE_RECORD("all2all_forward.start", local_gpu->get_stream(), false,
                   local_gpu->get_device_id());
    for (size_t j = 0; j < local_gpu_count; j++) {
      CK_NCCL_THROW_(ncclSend(src_pos[i][j], table[i][j], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
      CK_NCCL_THROW_(ncclRecv(dst_pos[i][j], table[j][i], type, j, local_gpu->get_nccl(),
                              local_gpu->get_stream()));
    }
    PROFILE_RECORD("all2all_forward.stop", local_gpu->get_stream(), false,
                   local_gpu->get_device_id());
  }
  CK_NCCL_THROW_(ncclGroupEnd());

  return;
}

template void SparseEmbeddingFunctors::all2all_forward<float>(
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, const Tensors2<float> &send_tensors, Tensors2<float> &recv_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::all2all_forward<__half>(
    size_t batch_size_per_gpu, const std::vector<size_t> &slot_num_per_gpu,
    size_t embedding_vec_size, const Tensors2<__half> &send_tensors, Tensors2<__half> &recv_tensors,
    const ResourceManager &resource_manager);

#endif

}  // namespace HugeCTR
