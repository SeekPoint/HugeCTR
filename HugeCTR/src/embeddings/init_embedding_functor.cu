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

#include "HugeCTR/include/data_simulator.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/general_buffer2.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {
/*
我们来分析 init_embedding_per_gpu，其实就是简单的用 % 运算来进行分配。
 举出一个例子来看看：假如10个slot，3个GPU，则slot ID是 0～9，
 GPU id是0～2。0~10 % 3 = 0,1,2,0,1,2,0,1,2,0，
 所以10个slot 被分配到3个GPU，分别是：
    GPU 0 ：0，3，6，9
    GPU 1 : 1，4，7，
    GPU 2 ：2，5，8，
所以，slot per gpu 是不相等的。
 */
void SparseEmbeddingFunctors::init_embedding_per_gpu(size_t gid, size_t total_gpu_count,
                                                     const std::vector<size_t> &slot_sizes,
                                                     size_t embedding_vec_size,
                                                     Tensors2<float> &embedding_tables,
                                                     Tensor2<size_t> &slot_ids,
                                                     const GPUResource &gpu_resource) {
  CudaDeviceContext context(gpu_resource.get_device_id());

  size_t *slot_ids_ptr = slot_ids.get_ptr();

  size_t key_offset = 0;
  size_t value_index_offset = 0;
  for (size_t i = 0, j = 0; i < slot_sizes.size(); i++) {  // 遍历slot
    size_t slot_size = slot_sizes[i];
    if ((i % total_gpu_count) == gid) {  // 本GPU id
      MESSAGE_("gpu" + std::to_string(gid) + " start to init embedding of slot" +
               std::to_string(i) + " , slot_size=" + std::to_string(slot_size) +
               ", key_offset=" + std::to_string(key_offset) +
               ", value_index_offset=" + std::to_string(value_index_offset));

      // 只有i等于gid时候，才会继续操作
      float up_bound = sqrt(1.f / slot_size);
      HugeCTR::UniformGenerator::fill(
          embedding_tables[j++], -up_bound, up_bound, gpu_resource.get_sm_count(),
          gpu_resource.get_replica_variant_curand_generator(), gpu_resource.get_stream());

      // 配置slot id
      memset_const(slot_ids_ptr, i, slot_size, gpu_resource.get_stream());

      value_index_offset += slot_size;
      slot_ids_ptr += slot_size;
    }
    key_offset += slot_size;
  }
}

}  // namespace HugeCTR
