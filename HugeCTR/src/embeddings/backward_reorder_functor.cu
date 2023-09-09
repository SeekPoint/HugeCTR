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

namespace {
/*
5.1 Reorder backward
Reorder反向传播目的就是让所有GPU之上的梯度被分散拷贝到 all2all_tensors_ 不同的位置。
 下图之中，每个slot对应一个梯度embedding vector，
 现在 train_output_tensors_(gradients) 之中是梯度。现在每个GPU之上的梯度都是一个完整的两个sample的梯度。
008-006
具体代码如下，这里每个GPU上都会有两个bid，分别对应了sample 1 和 sample 2：
 * */
// reorder operation before all2all in backward propagation
template <typename TypeEmbeddingComp>
__global__ void backward_reorder_kernel(int batch_size_per_gpu, int slot_num,
                                        int embedding_vec_size, int gpu_num,
                                        const TypeEmbeddingComp *input, TypeEmbeddingComp *output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding
  // vector gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one
  // sample on each GPU Each thread needs to process slot_num slots

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid;  // sample_id on the current GPU

  if ((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    // 源：本样本梯度的起始位置。GPU0是0，GPU1是1*4*embedding_vec_size
    int src_offset = sample_id * slot_num * embedding_vec_size;
    int src_stride = embedding_vec_size;  // 跨度。这里是4

    for (int slot_id = 0; slot_id < slot_num; slot_id++) {   // 取值是0～3
      int gpu_id = slot_id % gpu_num;  // 取值是0～1
      int offset_pre = 0;  // offset in previous gpus
      for (int id = 0; id < gpu_id; id++) {
        // 数值是2
        int slot_num_per_gpu = slot_num / gpu_num + ((id < (slot_num % gpu_num)) ? 1 : 0);
        // 数值是2
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        // 找到前面GPU之中，所有样本的起始位置，GPU0是0，GPU1是4
        offset_pre += stride;
      }

      // 目标位置：找到当前GPU之中，本样本的起始位置
      // slot_num_per_gpu = 2
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id < (slot_num % gpu_num)) ? 1 : 0);
      // 2*sample_id
      int offset_cur = sample_id * slot_num_per_gpu;  // offset in current gpu

      // 需要注意的是，embedding_vec_size 是4，但是在图上我们都把 embedding_vec_size 归结为一个slot
      // 如果对应到图上就是以slot为单位，embedding_vec_size就是1，所以简化如下：
      // GPU0=sample_id*2+0+slot_id/gpu_num，sample1是0～1，sample2是4～5
      // GPU1=sample_id*2+4+slot_id/gpu_num，sample1是2～3，sample2是6～7
      int dst_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      // 源位置：找到当前梯度之中，本样本的起始位置
      // 需要注意的是，embedding_vec_size 是4，但是在图上我们都把 embedding_vec_size 归结为一个slot
      // 如果对应到图上就是以slot为单位，embedding_vec_size就是1，所以简化如下：
      // src_offset=sample_id * slot_num
      // src_addr = sample_id * slot_num + slot_id
      // 则src_addr应该是：sample_id * slot_num + slot_id
      // 所以，GPU0，GPU1的取值范围都是sample1=0～3，sample2=4～7
      int src_addr = src_offset + src_stride * slot_id;
      output[dst_addr + tid] = input[src_addr + tid];  // 把本样本的梯度拷贝到 all2all_tensors_ 张量上应在的位置
    }
  }
}

// reorder operation before all2all in backward propagation
__global__ void backward_reorder_align2_kernel(int batch_size_per_gpu, int slot_num,
                                               int embedding_vec_size, int gpu_num,
                                               const __half *input, __half *output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding
  // vector gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one
  // sample on each GPU Each thread needs to process slot_num slots

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int sample_id = bid;  // sample_id on the current GPU

  if ((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    const __half2 *input2 = reinterpret_cast<const __half2 *>(input);
    __half2 *output2 = reinterpret_cast<__half2 *>(output);

    int src_offset = sample_id * slot_num * embedding_vec_size;
    int src_stride = embedding_vec_size;

    for (int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num;
      int offset_pre = 0;  // offset in previous gpus
      for (int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id < (slot_num % gpu_num)) ? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;
      }
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id < (slot_num % gpu_num)) ? 1 : 0);
      int offset_cur = sample_id * slot_num_per_gpu;  // offset in current gpu
      int dst_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int src_addr = src_offset + src_stride * slot_id;
      output2[dst_addr + tid] = input2[src_addr + tid];
    }
  }
}

template <typename TypeEmbeddingComp>
void do_backward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                         size_t total_gpu_count, const TypeEmbeddingComp *input,
                         TypeEmbeddingComp *output, cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  const size_t block_size = embedding_vec_size;
  backward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
}

void do_backward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                         size_t total_gpu_count, const __half *input, __half *output,
                         cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    backward_reorder_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size / 2, total_gpu_count, input, output);
  } else {
    const size_t block_size = embedding_vec_size;
    backward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
  }
}

}  // namespace

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward_reorder(size_t batch_size_per_gpu, size_t slot_num,
                                               size_t embedding_vec_size,
                                               const Tensors2<TypeEmbeddingComp> &src_tensors,
                                               Tensors2<TypeEmbeddingComp> &dst_tensors,
                                               const ResourceManager &resource_manager) {
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  backward_reorder(batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, src_tensors,
                   dst_tensors, resource_manager);
}

template void SparseEmbeddingFunctors::backward_reorder<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<float> &src_tensors, Tensors2<float> &dst_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward_reorder<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<__half> &src_tensors, Tensors2<__half> &dst_tensors,
    const ResourceManager &resource_manager);

template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::backward_reorder(size_t batch_size_per_gpu, size_t slot_num,
                                               size_t embedding_vec_size, size_t total_gpu_count,
                                               const Tensors2<TypeEmbeddingComp> &src_tensors,
                                               Tensors2<TypeEmbeddingComp> &dst_tensors,
                                               const ResourceManager &resource_manager) {
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  CudaDeviceContext context;
  for (size_t id = 0; id < local_gpu_count; id++) {
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());

    do_backward_reorder(batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
                        src_tensors[id].get_ptr(), dst_tensors[id].get_ptr(),
                        local_gpu->get_stream());
  }
}

template void SparseEmbeddingFunctors::backward_reorder<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size, size_t total_gpu_count,
    const Tensors2<float> &src_tensors, Tensors2<float> &dst_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::backward_reorder<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size, size_t total_gpu_count,
    const Tensors2<__half> &src_tensors, Tensors2<__half> &dst_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR
