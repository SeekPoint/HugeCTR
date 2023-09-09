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
4.3.1 思路
具体逻辑是：
  gpu_num 是全局有多少个GPU，后面也是想依据全局信息来计算，因为 all2all之后已经是一个全局视角了。

  拿到当前样本在当前GPU的sample id（其实就是bid，每个bid对应一个sample），
  后面都是针对这个sample id进行处理，这样能保证只保留本GPU的sample。
  比如第2个sample，则sample_id = 1。

  拿到当前样本的第一个slot的起始位置，比如 1 * 4 * 8 = 32。

  得到一个slot对应的embedding vector的大小，就是slot和slot之间的stride = 8

  遍历sample的slots，范围是0~slot num，目的是从 all2all 之中拷贝这些slots到embedding_data_.get_output_tensors，
  所以需要找到本sample的slot在all2all的起始位置。

  对于每个slot，需要找到slot在哪个gpu之上。
        遍历GPU，遍历GPU的目的是，因为slot是按照GPU分配的，所以找前面GPU的位置，其实就是找前面slot的位置。
        offset_pre 最终得到的就是在本slot之前的GPU之上有多少个slots。
                 这里关键代码是 gpu_id = slot_id % gpu_num，这个用来确定“在哪个GPU传来的buffer之上找到某个slot”。

                针对我们例子，alltoall发送时候，是2个slot一起发送，这里reorder则需要一个slot一个slot的进行寻找数据，
                此时gpu_id就是用来寻找的关键点。

       得到每个GPU对应几个slot。
       得到当前sample在当前GPU的offset。
       得到当前sample在其他slot对应的数据起始位置。
       得到当前slot在 embedding_data_.get_output_tensors 之中的目标位置。
       拷贝本sample对应的第slot_id的信息。

4.3.2 图示
这里是为了演示，把逻辑简化了，
embedding_feature_tensors_, all2all_tensors_ 本来应该是一维数组，
这里抽象成了二维数组。
008-005
 * */
// reorder operation after all2all in forward propagation
template <typename TypeEmbeddingComp>
__global__ void forward_reorder_kernel(int batch_size_per_gpu, int slot_num, int embedding_vec_size,
                                       int gpu_num, const TypeEmbeddingComp *input,
                                       TypeEmbeddingComp *output) {
  // blockDim.x = embedding_vec_size; // each thread corresponding to one element of embedding
  // vector gridDim.x = batch_size / gpu_num = samples_per_gpu; // each block corresponding to one
  // sample on each GPU Each thread needs to process slot_num slots

  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // 当前GPU的sample id，后面都是针对这个sample id进行处理，这样能保证只保留本GPU的sample
  int sample_id = bid;  // sample_id on the current GPU

  if ((bid < batch_size_per_gpu) && (tid < embedding_vec_size)) {
    // 当前样本的第一个slot的起始位置，比如 1 * 4 * 8 = 32
    int dst_offset =
        sample_id * slot_num * embedding_vec_size;  // offset for the first slot of one sample
    // 一个slot对应的embedding vector的大小，就是slot和slot之间的stride = 8
    int dst_stride = embedding_vec_size;            // stride from slot to slot

    // 遍历sample的slots，范围是0~slot num，目的是从 all2all 之中拷贝这些slots到embedding_data_.get_output_tensors
    // 所以需要找到本sample的slot在all2all的起始位置
    for (int slot_id = 0; slot_id < slot_num; slot_id++) {
      int gpu_id = slot_id % gpu_num;  // 关键代码，确定slot在哪个gpu之上
      int offset_pre = 0;  // offset in previous gpus

      // 遍历GPU的目的是，因为slot是按照GPU分配的，所以找前面GPU的位置，其实就是找前面slot的位置
      // offset_pre 最终得到的就是在本slot之前的GPU之上有多少个slots
      for (int id = 0; id < gpu_id; id++) {
        int slot_num_per_gpu = slot_num / gpu_num + ((id < (slot_num % gpu_num)) ? 1 : 0);
        int stride = batch_size_per_gpu * slot_num_per_gpu;
        offset_pre += stride;  // 找到前面的位置
      }

      // 每个GPU对应几个slot
      int slot_num_per_gpu = slot_num / gpu_num + ((gpu_id < (slot_num % gpu_num)) ? 1 : 0);

      // 当前sample在当前GPU的offset
      int offset_cur = sample_id * slot_num_per_gpu;  // offset in current gpu

      // 当前sample在其他slot对应的数据起始位置
      // (offset_cur + offset_pre + (int)(slot_id / gpu_num))就是本slot前面有多少个slot
      int src_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      // 当前slot在 embedding_data_.get_output_tensors 之中的目标位置
      int dst_addr = dst_offset + dst_stride * slot_id;

      // 拷贝本sample对应的第slot_id的信息
      output[dst_addr + tid] = input[src_addr + tid];
    }
  }
}

// reorder operation after all2all in forward propagation
__global__ void forward_reorder_align2_kernel(int batch_size_per_gpu, int slot_num,
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

    int dst_offset =
        sample_id * slot_num * embedding_vec_size;  // offset for the first slot of one sample
    int dst_stride = embedding_vec_size;            // stride from slot to slot

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
      int src_addr = (offset_cur + offset_pre + (int)(slot_id / gpu_num)) * embedding_vec_size;

      int dst_addr = dst_offset + dst_stride * slot_id;
      output2[dst_addr + tid] = input2[src_addr + tid];
    }
  }
}

//do_forward_reorder 代码如下，其是依靠 forward_reorder_kernel 完成具体逻辑。
template <typename TypeEmbeddingComp>
void do_forward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        size_t total_gpu_count, const TypeEmbeddingComp *input,
                        TypeEmbeddingComp *output, cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  const size_t block_size = embedding_vec_size;
  forward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
}

void do_forward_reorder(size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
                        size_t total_gpu_count, const __half *input, __half *output,
                        cudaStream_t stream) {
  const size_t grid_size = batch_size_per_gpu;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_reorder_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size / 2, total_gpu_count, input, output);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_reorder_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count, input, output);
  }
}

}  // namespace

/**
 * reoder the sequence of data after all2all operation in forward propagation
 * @param batch_size_per_gpu batch size per GPU
 * @param slot_num the number of localized slots
 * @param embedding_vec_size embedding vector size.
 * @param src_tensors the source tensors before reorder
 * @param dst_tensors the destination tensors after reorder
 * @param device_resources all gpus device resources.
 * @param context gpu device context, for switching device.
 */
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_reorder(size_t batch_size_per_gpu, size_t slot_num,
                                              size_t embedding_vec_size,
                                              const Tensors2<TypeEmbeddingComp> &src_tensors,
                                              Tensors2<TypeEmbeddingComp> &dst_tensors,
                                              const ResourceManager &resource_manager) {
  CudaDeviceContext context;
  size_t local_gpu_count = resource_manager.get_local_gpu_count();
  size_t total_gpu_count = resource_manager.get_global_gpu_count();
  forward_reorder<TypeEmbeddingComp>(batch_size_per_gpu, slot_num, embedding_vec_size,
                                     total_gpu_count, src_tensors, dst_tensors, resource_manager);
}

template void SparseEmbeddingFunctors::forward_reorder<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<float> &src_tensors, Tensors2<float> &dst_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::forward_reorder<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size,
    const Tensors2<__half> &src_tensors, Tensors2<__half> &dst_tensors,
    const ResourceManager &resource_manager);

/*
4.3 Reorder
我们可以发现，现在每个GPU之上都拥有自己的数据（每个GPU都是一个完整的sample），
但是sample数据内部顺序有点问题，不是按照slot升序，
我们把上图再大致调整细化一下（图例与实际变量有出入，这里只是为了更好的演示）。
008-004
接下来使用 Reorder 从 all2all_tensor 拷贝到 embedding_data_.get_output_tensors(is_train)，
在拷贝过程中选择会调整顺序，目的是把 slot 0, slot 2, slot 1 , slot 3 转换为 slot 0, slot 1, slot 2, slot3。
 * */
template <typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_reorder(size_t batch_size_per_gpu, size_t slot_num,
                                              size_t embedding_vec_size, size_t total_gpu_count,
                                              const Tensors2<TypeEmbeddingComp> &src_tensors,
                                              Tensors2<TypeEmbeddingComp> &dst_tensors,
                                              const ResourceManager &resource_manager) {
  CudaDeviceContext context;
  size_t local_gpu_count = resource_manager.get_local_gpu_count();

  for (size_t id = 0; id < local_gpu_count; id++) { // 遍历本地GPU
    const auto &local_gpu = resource_manager.get_local_gpu(id);
    context.set_device(local_gpu->get_device_id());

    // 拷贝
    do_forward_reorder(batch_size_per_gpu, slot_num, embedding_vec_size, total_gpu_count,
                       src_tensors[id].get_ptr(), dst_tensors[id].get_ptr(),
                       local_gpu->get_stream());
  }
}

template void SparseEmbeddingFunctors::forward_reorder<float>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size, size_t total_gpu_count,
    const Tensors2<float> &src_tensors, Tensors2<float> &dst_tensors,
    const ResourceManager &resource_manager);

template void SparseEmbeddingFunctors::forward_reorder<__half>(
    size_t batch_size_per_gpu, size_t slot_num, size_t embedding_vec_size, size_t total_gpu_count,
    const Tensors2<__half> &src_tensors, Tensors2<__half> &dst_tensors,
    const ResourceManager &resource_manager);

}  // namespace HugeCTR
