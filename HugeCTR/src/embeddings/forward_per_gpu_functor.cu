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
#include "HugeCTR/include/utils.cuh"

namespace HugeCTR {

namespace {
/**
 * All the CUDA kernel functions used by embedding layer are defined in this file, including
 * forward propagation, backward propagation. The functions are defined by propagation type
 * and combiner type(sum or mean) as below:
 *   1) forward
 *        sum: calling forward_sum_kernel()
 *        mean: calling foward_sum_kernel() + forward_scale_kernel()
 *   2) backward:
 *        calculating wgrad:
 *          sum: calling backward_sum_kernel()
 *          mean: calling backward_mean_kernel()
 *        update embedding table: including several steps as below,
 *          step1: expand sample IDs, calling sample_id_expand_kernel()
 *          step2: get value_index by key (will call hash_table->get_insert() in nv_hashtable lib)
 *          step3: sort by value_index (will call cub::DeviceRadixSort::SortPairs in cub lib)
 *          step4: count the number for each unduplicated value_index, calling value_count_kernel()
 *          step5: use optimizer method to compute deltaw, and record corresponding, including three
 * types of optimizer: Adam: caling opt_adam_kernel() Momentum sgd: calling
 * opt_momentum_sgd_kernel() Nesterov: calling opt_nesterov_kernel() step6: update embedding table
 * by deltaw, calling update_kernel()
 */
/*
4.3.3.1 例子
回忆我们的例子：
*   40,50,10,20
*   30,50,10
*   30,20
*   10
* Will be convert to the form of:
* row offset: 0,4,7,9,10
* value: 40,50,10,20,30,50,10,30,20,10
第一个样本包括：
    40,50,10,20 # slot 1
    30,50,10 # slot 2
第二个样本是
    30,20 # slot 1
    10 # slot 2
所以，应该得到10个稠密向量，比如40有一个稠密向量，50有一个稠密向量。
怎么知道 40 对应低维嵌入表的哪一行呢？通过一个哈希表来处理的，假如哈希函数是选取十位数为key，则得到：
        m_hf(40)=4
所以，就知道了，40应该在低维嵌入表的第4行（我们对哈希表做了简化）。
4.3.3.2 要点
forward_sum_kernel 的代码如下，这里代码很烧脑，需要结合注释仔细分析，
第一个要点是回忆一下hash_value_index_tensors_的使用：
细心读者可能有疑问，如果哈希表能从高维offset映射到低维offset，这个hash_value_index_tensors_ 应该就没有用了吧？这里解释如下：
    事实上，因为解耦合的原因，hash_value_index_tensors_ 并不应该知道 哈希表内部把高维矩阵的维度映射了多大的低维矩阵，而 hash_value_index_tensors_ 大小也不应该随之变化。
    所以，hash_value_index_tensors_ 大小被固定为：batch_size * nnz_per_slot，可以认为就是CSR之中元素个数。所以 hash_value_index_tensors_ 实际上记录了每个元素对应的低维矩阵offset 数值，hash_value_index_tensors_ 事实上就是和CSR之中元素位置一一对应。
    因此，最终嵌入表查找时候，是通过CSR row offset 来找到 CSR之中每个元素，从而也找到了hash_value_index_tensors_ 这个表的index，从而就能找到其低维矩阵offset。
    针对我们的例子，hash_value_index_tensors_ 的数值就是 4,5,1,2,3,5,1,3,2,1。
其余几个要点是：
    bid 是第几个样本。
    tid 是最终嵌入向量的第几个元素，一个线程处理嵌入向量的一个元素。
    hash_value_index 是低维嵌入表的offset表的指针。
        hash_value_index 是一张表，就是上面说的hash_value_index_tensors_。
    row_offset 是CSR offset，例子就是 0,4,7,9,10，所以对于第二个样本，row offset 是 7，9。
    hash_table_value 可以认为是一个数组，低维嵌入矩阵是存储在这个数组之中。hash_table_value[value_index * embedding_vec_size] 就是对应的稠密向量。
4.3.3.3 注释版代码
..
4.3.3.4 并行操作
关于并行操作，留意点是：
    bid是第几个样本。
    tid 是最终嵌入向量的第几个元素，一个线程处理嵌入向量的一个元素。
    hash_table_value[value_index * embedding_vec_size] 就是 CSR user ID对应的稠密向量。
    hash_table_value[value_index * embedding_vec_size + tid] 就是 CSR user ID对应的稠密向量的第 tid 个element。
之前说了，应该是两个样本一共10个元素 40,50,10,20,30,50,10,30,20,10，应该对应10个稠密向量。
 但是在GPU之中会启动tid个线程并行操作，会在一个slot之中进行reduce，
 然后把结果存入到 embedding_feature 之中。GPU并行体现在同时生成一个稠密向量的所有元素。
 就是每一个sample生成 slot_num 个稠密向量。稠密向量的每个元素都是根据样本内部元素计算出来的。

比如第一个样本是：
        40,50,10,20 # slot 1
        30,50,10 # slot 2
    slot 1 应该输出 40 对应的稠密向量 + 50 对应的稠密向量 + 10 对应的稠密向量 + 20 对应的稠密向量。
    slot 2 应该输出 30 对应的稠密向量 + 50 对应的稠密向量 + 10 对应的稠密向量。
但经过 combiner之后，样本1输出了两个稠密向量，分别对应两个slot，假定每个稠密向量长度是8，计算方式是：
稠密向量1 = 40 对应的稠密向量 + 50 对应的稠密向量 + 10 对应的稠密向量 + 20 对应的稠密向量
稠密向量2 = 30 对应的稠密向量 + 50 对应的稠密向量 + 10 对应的稠密向量
稠密向量1内部8个元素分别是由40，50，10，20对应的稠密向量8个同位置上元素的和构成。
 即 稠密向量1的[0] = sum(40 对应的稠密向量的[0], 50 对应的稠密向量的[0], 10 对应的稠密向量的[0], 20 对应的稠密向量的[0] )。
 可以看到，其确实转成了嵌入式向量，但并不是用矩阵乘法，而是用了自己一套机制，具体入下图：
006-007.jpg
 */
// forward kernel funcion: for both combiner=sum and combiner=mean
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_sum_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                   const TypeKey *row_offset, const size_t *hash_value_index,
                                   const float *hash_table_value,
                                   TypeEmbeddingComp *embedding_feature) {
  // bid是第几个样本，假如是1，那么就是第二个样本
  int bid = blockIdx.x;   // each block corresponding to one sample
  // tid最终是嵌入向量的第几个元素，一个线程处理嵌入向量的一个元素
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {  // batch_size = 2
    for (int i = 0; i < slot_num; i++) {  // slot_num = 2
      // 得到当前行对应的在row offset之中的位置，比如是2或者3，就是从 0,4,7,9,10 之中找第2，第3个
      int feature_row_index = bid * slot_num + i;   // feature_row_index 范围是 2,3
      // 得到当前行在CSR内的元素偏移，行0，行1 是第一个样本，行2，行3是第二个样本
      TypeKey value_offset = row_offset[feature_row_index];  // 行2的偏移value_offset是7，行3是9
      // 每行有多少元素，行2对应的元素个数是9-7=2，行3对应的元素个数是10-9=1
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; j++) { // 行内元素个数，行2是2，行3是1
        // 假如是行2，则value是30，20，则取出hash_value_index的第7，8个位置的数值，分别是3，2
        size_t value_index = hash_value_index[value_offset + j];

        // 取出hash_table_value的第3，2个元素的数值，进行计算
        // value_index 就是具体哪一个 CSR user ID 在 hash_table_value 之中的起始位置，即hash_value_index记录了哪一个 CSR user ID 在hash_table_value的第几行
        // hash_table_value[value_index * embedding_vec_size] 就是 CSR user ID对应的稠密向量
        // hash_table_value[value_index * embedding_vec_size + tid] 就是 CSR user ID对应的稠密向量的第tid个element
        sum += (value_index != std::numeric_limits<size_t>::max())
                   ? hash_table_value[value_index * embedding_vec_size + tid]
                   : 0.0f;
      }

      // store the embedding vector
      // 这里就对应了2，3两行，就是一个样本的两个slots 会顺序排在一起，最终输出稠密向量的每个元素值是样本之中所有元素稠密向量对应位置元素的和
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum);
    }
  }
}

template <typename TypeKey>
__global__ void forward_sum_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                          const TypeKey *row_offset, const size_t *hash_value_index,
                                          const float *hash_table_value,
                                          __half *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    const float2 *hash_table_value2 = reinterpret_cast<const float2 *>(hash_table_value);
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      TypeKey feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      // use float type to do accumulation
      float2 sum2 = {0.0f, 0.0f};
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum2.x += (value_index != std::numeric_limits<size_t>::max())
                      ? hash_table_value2[value_index * embedding_vec_size + tid].x
                      : 0.0f;
        sum2.y += (value_index != std::numeric_limits<size_t>::max())
                      ? hash_table_value2[value_index * embedding_vec_size + tid].y
                      : 0.0f;
      }
      __half2 sum = __float22half2_rn(sum2);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = sum;
    }
  }
}

// forward kernel funcion: for combiner=mean in LocalizedEmbedding
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void forward_mean_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                    const TypeKey *row_offset, const size_t *hash_value_index,
                                    const float *hash_table_value,
                                    TypeEmbeddingComp *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      int feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      float sum = 0.0f;

      // reduce in a slot
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum += (value_index != std::numeric_limits<size_t>::max())
                   ? hash_table_value[value_index * embedding_vec_size + tid]
                   : 0.0f;
      }

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / feature_num;
      }

      // store the embedding vector
      embedding_feature[feature_row_index * embedding_vec_size + tid] =
          TypeConvertFunc<TypeEmbeddingComp, float>::convert(sum * scaler);
    }
  }
}

template <typename TypeKey>
__global__ void forward_mean_align2_kernel(int batch_size, int slot_num, int embedding_vec_size,
                                           const TypeKey *row_offset,
                                           const size_t *hash_value_index,
                                           const float *hash_table_value,
                                           __half *embedding_feature) {
  int bid = blockIdx.x;   // each block corresponding to one sample
  int tid = threadIdx.x;  // each thread corresponding to one element in the embedding vector

  if (bid < batch_size && tid < embedding_vec_size) {
    const float2 *hash_table_value2 = reinterpret_cast<const float2 *>(hash_table_value);
    __half2 *embedding_feature2 = reinterpret_cast<__half2 *>(embedding_feature);

    for (int i = 0; i < slot_num; i++) {
      int feature_row_index = bid * slot_num + i;
      TypeKey value_offset = row_offset[feature_row_index];
      int feature_num =
          row_offset[feature_row_index + 1] - value_offset;  // number of hash values in one slot

      // use float to do accumulation
      float2 sum = {0.0f, 0.0f};
      for (int j = 0; j < feature_num; j++) {
        size_t value_index = hash_value_index[value_offset + j];
        sum.x += (value_index != std::numeric_limits<size_t>::max())
                     ? hash_table_value2[value_index * embedding_vec_size + tid].x
                     : 0.0f;
        sum.y += (value_index != std::numeric_limits<size_t>::max())
                     ? hash_table_value2[value_index * embedding_vec_size + tid].y
                     : 0.0f;
      }
      __half2 sum2 = __float22half2_rn(sum);

      float scaler = 1.0f;
      if (feature_num > 1) {
        scaler = 1.0f / feature_num;
      }
      __half2 scaler2 = __float2half2_rn(scaler);

      // store the embedding vector
      embedding_feature2[feature_row_index * embedding_vec_size + tid] = __hmul2(sum2, scaler2);
    }
  }
}

// do sum reduction
template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                 cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  const size_t block_size =
      embedding_vec_size;  // each thread corresponds to one element in an embedding vector
  forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(batch_size, slot_num, embedding_vec_size,
                                                           row_offset, hash_value_index,
                                                           hash_table_value, embedding_feature);
}

// do sum reduction
template <typename TypeHashKey>
void forward_sum(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                 const TypeHashKey *row_offset, const size_t *hash_value_index,
                 const float *hash_table_value, __half *embedding_feature, cudaStream_t stream) {
  const size_t grid_size = batch_size;  // each block corresponds to a sample
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_sum_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
        hash_table_value, embedding_feature);
  } else {
    const size_t block_size =
        embedding_vec_size;  // each thread corresponds to one element in an embedding vector
    forward_sum_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
        embedding_feature);
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, TypeEmbeddingComp *embedding_feature,
                  cudaStream_t stream) {
  const size_t grid_size = batch_size;
  const size_t block_size = embedding_vec_size;
  forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
      batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
      embedding_feature);
}

template <typename TypeHashKey>
void forward_mean(size_t batch_size, size_t slot_num, size_t embedding_vec_size,
                  const TypeHashKey *row_offset, const size_t *hash_value_index,
                  const float *hash_table_value, __half *embedding_feature, cudaStream_t stream) {
  const size_t grid_size = batch_size;
  if (embedding_vec_size % 2 == 0) {
    const size_t block_size = embedding_vec_size / 2;
    forward_mean_align2_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size / 2, row_offset, hash_value_index,
        hash_table_value, embedding_feature);
  } else {
    const size_t block_size = embedding_vec_size;
    forward_mean_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, embedding_vec_size, row_offset, hash_value_index, hash_table_value,
        embedding_feature);
  }
}

}  // namespace


/*
4.2 查找
目前代码来到了这里，就是利用哈希表来从输入CSR得到对应的嵌入向量。
006-004.jpg
forward_per_gpu 分为两部分：查找和内部规约。

4.2.1 查找算子
    @forward_per_gpu 函数是用来具体做lookup的。从其注释可以看到其用途，就是我们之前分析过的。
    @param row_offset row_offset (CSR format of input sparse tensors)
    @param hash_key value (CSR format of input sparse tensors)
    @param nnz non-zero feature number per batch
    @param hash_table hash table, pairs of <key, value_index>
    @param hash_table_value hash table value, which represents embedding vector
    @param hash_value_index hash table value_index(row index of embedding)
这里的参数都是引用，可以修改外部数据，具体思路是：
    首先使用 hash_key value (CSR format of input sparse tensors) 来调用 get_insert 去 hash table 之中查找，如果找到了，得到的就是 hash_value_index。
    这个value 是 低维 embedding表 的 row index。这部分代码是 hash_table.get_insert 相关。其实，这里没有用到get_insert 返回值，
    而是把 hash_key value 插进哈希表内部，得到一个映射，具体如何查找是通过 csr row offset完成。

    hash_table.get_insert 如果在 hash_table 的内部数据结构之中找到了，就返回，
    如果没有找到，就插入一个递增的数值，这个数值被设置到 hash_value_index 之中。

    然后通过 hash_value_index 作为 index，在 hash_table_value 之中得到最终的 embedding vector，并且先在slot内部做reduce。
    这部分代码是 forward_sum 和 forward_mean 相关。

所以 hash_table_value_tensors_[i], hash_value_index_tensors_ 这两部分何时设置？
其实是在forward_per_gpu完成的，具体逻辑如图：
        006-005.jpg
具体代码是：
*/
/**
 * forward propagation on each GPU for LocalizedSlotSparseEmbeddingHash
 * @param batch_size batch size for the current mini-batch computation.
 * @param slot_num the number of slots for current GPU
 * @param embedding_vec_size embedding vector size.
 * @param combiner 0-sum; 1-mean
 * @param row_offset row_offset (CSR format of input sparse tensors)
 * @param hash_key value (CSR format of input sparse tensors)
 * @param nnz non-zero feature number per batch
 * @param hash_table hash table, pairs of <key, value_index>
 * @param hash_table_value hash table value, which represents embedding vector
 * @param hash_value_index hash table value_index(row index of embedding)
 * @param embedding_feature embedding feature (output)
 * @param stream cuda stream
 3.3 调用
为了更好的分析，在看 concurrent_unordered_map 之前，我们需要看看如何调用HashTable。
 调用代码是HugeCTR/src/embeddings/forward_per_gpu_functor.cu 之中的forward_per_gpu方法。这里已经是 CUDA 代码了。
 */
template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_per_gpu(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<TypeHashKey> &row_offset, const Tensor2<TypeHashKey> &hash_key, size_t nnz,
    HashTable<TypeHashKey, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<TypeEmbeddingComp> &embedding_feature,
    cudaStream_t stream) {
  try {
    if (train) {  // 训练会来到这里
      // 这里会调用插入代码// 先从hash_table之中依据 hash_key 得到hash_value_index 之中对应的位置，
      // 作用就是让 hash_value_index 之中包含所有key对应的内部hash_value_index
      // 其实，这里是否返回不重要，重要的是把hash_key value插进哈希表内部，具体如何查找是通过csr row offset完成
      /*
算子内部也分为 get_insert 来处理哈希表，和 combiner 处理，我们一一看看。
      4.2.2 get_insert
前面我们分析了哈希表的 get 和 insert 操作，这里是合而为一，就是如果找不到就插入。
开始训练时候，不需要给哈希表赋初值，而是在训练过程之中使用get_insert动态插入。
我们再回忆下原理。
比如一共有1亿个单词，40表示第40个单词。如果想表示 10，30，40，50，20在这一亿个单词是有效的，
最常见的办法是弄个1亿长度数组，把40，50，20，30，10这5个位置设置为1，其他位置设置为0。
对应嵌入矩阵也是一个高维矩阵，比如 1亿 x 64 维度矩阵。
如果想省空间，就弄会构建一个小数据结构（低维矩阵）来存储这些有意义的值，弄一个一个hash函数 m_hf来做"从高维矩阵到低维矩阵的转换"，就是10 -->?，20 --> ? 等。
假如是选取十位数数为key，对于我们的例子，就是
m_hf(10)=1
m_hf(20)=2
m_hf(30)=3
m_hf(40)=4
m_hf(50)=5
1，2，3，4，5 就是内部的hash_value，叫做 hash_value（对应下面代码），对应的内部存储数组叫做 hashtbl_values。
但是因为分桶了，所以在哈希表内部是放置在hashtbl_values之中（这里我们做了一个简化，就是 hashtbl_values[i] = i）。
  hashtbl_values[1] = 1，hashtbl_values[2] = 2, hashtbl_values[3] = 3，...
以上说的是哈希表，我们回到 DistributedSlotSparseEmbeddingHash 本身，
于是1，2，3 （数组之中的内容，不是数组index，简化成恰好相等）就是DistributedSlotSparseEmbeddingHash 想得到的 10, 20, 30 对应的数据，
就是10 放在低维嵌入表第一个位置，20放在低维嵌入表第二个位置，就是就是低维矩阵的row offset。
即，hash_value_index 的内容是 [1,2,3,4,5]，这些是原始输入数据 10，20，30，40，50 分别在 hash_table_value 之中对应的index，
因此，10 对应的就是 hash_table_value[1]，20 对应就是 hash_table_value[2]，依此类推。
       * */
      hash_table.get_insert(hash_key.get_ptr(), hash_value_index.get_ptr(), nnz, stream);
    } else {
      hash_table.get_mark(hash_key.get_ptr(), hash_value_index.get_ptr(), nnz, stream);
    }
    /*5.1.2.2 Pooling
    具体如何做pooling？HugeCTR有sum或者mean两种操作，具体叫做combiner，比如：

     结合前面图，类别特征就一共有M个slot，对应了M个嵌入表。 004-015.png

     比如在 test/pybind_test/din_matmul_fp32_1gpu.py 之中，可见CateID有11个slots。

      model.add(hugectr.Input(label_dim = 1, label_name = "label",
                             dense_dim = 0, dense_name = "dense",
                             data_reader_sparse_param_array =
                             [hugectr.DataReaderSparseParam("UserID", 1, True, 1),
                             hugectr.DataReaderSparseParam("GoodID", 1, True, 11),
                             hugectr.DataReaderSparseParam("CateID", 1, True, 11)]))
      比如，下图之中，004-016.png
      一个sample有7个key，分为两个field，就是两个slot。4个key放在第一个slot之上，3个key放到第二个slot上，第三个slot没有key。
      在查找过程中，会把这些key对应的value查找出来。第一个slot内部会对这些value进行sum或者mean操作得到V1，
           第二个slot内部对3个value进行sum或者mean操作得到V2，最后会把V1，V2进行concat操作，传送给后续层。
*/
/*
4.3 combiner
 拿到了多个向量之后，需要做聚合，因为此处过于繁琐，因此我们单独拿出来说一下，把它提升到和查找一个级别，大家不要误会。
4.3.1 为何要聚合
 在CTR领域，人们通常会把多个embedding向量合并成一个向量，这就是pooling。
 比如用户看了3本艺术书，2本体育书，所以 读书习惯 = 3 * 艺术 + 2 * 体育。这种聚合经常使用加权的pooling，而不是concat。
 因为虽然concat效果更好，但是pooling更快，而且这样做好处就是即使向量长度不同，也可以生成一个同样长度的新张量。
 比如：特征的embeddingSize是10，现在所有Field的个数是50，其中5个Field是序列形式的特征（对于序列长度的上限取40）。此时你有两种处理方式：
     mean/sum pooling ：embedding层的参数量是10 * 50 = 500
     concat ：embedding层的参数量是 10*(50-5) + 40 * 10 * 5 = 2450
 如果使用 concat，则embedding层的参数量直接涨了4倍左右，实际ctr模型种参数量最大部分一般就是embedding -> MLP的这一层，所以concat会直接拖慢线上inference的速度。
4.3.2 设计准则
 我们回忆一下前面提到的设计准则：嵌入表可以被分割成多个槽（或feature fields）。为了在不同的嵌入上获得最佳性能，可以选择不同的嵌入层实现。
      LocalizedSlotEmbeddingHash：同一个槽（特征域）中的特征会存储在一个GPU中，这就是为什么它被称为“本地化槽”，根据槽的索引号，不同的槽可能存储在不同的GPU中。
      DistributedSlotEmbeddingHash：所有特征都存储于不同特征域/槽上，不管槽索引号是多少，这些特征都根据特征的索引号分布到不同的GPU上。这意味着同一插槽中的特征可能存储在不同的 GPU 中，这就是将其称为“分布式插槽”的原因。由于需要全局规约，所以DistributedSlotEmbedding 适合 embedding 大于 GPU 内存大小的情况，因而DistributedSlotEmbedding 在 GPU 之间有更多的内存交换。
一定要注意，LocalizedSlotEmbeddingHash 和 DistributedSlotEmbeddingHash 的区别在于同一个槽（特征域）中的特征 是不是 会存储在同一个GPU中。比如，有 2 张GPU卡，有4个slot。
      local模式 ：GPU0存slot0和slot1，GPU1存slot2和slot3。
      distribute模式 ：每个GPU都会存所有slot的一部分参数，通过哈希方法决定如何将一个参数分配到哪个GPU上。
在嵌入查找过程中，属于同一槽的稀疏特征输入在分别转换为相应的密集嵌入向量后，被简化为单个嵌入向量。然后，来自不同槽的嵌入向量连接在一起。这个就是前面提到的combiner操作。

4.3.3 Combiner代码
现在已经拿到了 embedding table 的 index，需要看看如何拿到 embedding vector，如何仅需操作。
     具体是通过 forward_sum 和 forward_mean 完成，我们用 forward_sum 举例看看
 * */

    // do sum reduction
    if (combiner == 0) {  // 0-sum; 1-mean
      // 然后利用 hash_value_index 从 hash_table_value 之中得到 value，再进行操作
      forward_sum(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                  hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                  embedding_feature.get_ptr(), stream);
    } else if (combiner == 1) {
      // 然后利用 hash_value_index 从 hash_table_value 之中得到 value，再进行操作
      forward_mean(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                   hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                   embedding_feature.get_ptr(), stream);
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::forward_per_gpu<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    HashTable<unsigned int, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    HashTable<long long, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    HashTable<unsigned int, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<__half> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_per_gpu<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    HashTable<long long, size_t> &hash_table, const Tensor2<float> &hash_table_value,
    Tensor2<size_t> &hash_value_index, Tensor2<__half> &embedding_feature, cudaStream_t stream);

template <typename TypeHashKey, typename TypeEmbeddingComp>
void SparseEmbeddingFunctors::forward_sum_per_gpu(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<TypeHashKey> &row_offset, const Tensor2<TypeHashKey> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<TypeEmbeddingComp> &embedding_feature, cudaStream_t stream) {
  try {
    // do sum reduction
    if (combiner == 0) {
      forward_sum(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                  hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                  embedding_feature.get_ptr(), stream);
    } else if (combiner == 1) {
      forward_mean(batch_size, slot_num, embedding_vec_size, row_offset.get_ptr(),
                   hash_value_index.get_ptr(), hash_table_value.get_ptr(),
                   embedding_feature.get_ptr(), stream);
    } else {
      CK_THROW_(Error_t::WrongInput, "Invalid combiner type ");
    }
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template void SparseEmbeddingFunctors::forward_sum_per_gpu<unsigned int, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<long long, float>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<float> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<unsigned int, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<unsigned int> &row_offset, const Tensor2<unsigned int> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<__half> &embedding_feature, cudaStream_t stream);

template void SparseEmbeddingFunctors::forward_sum_per_gpu<long long, __half>(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size, int combiner, bool train,
    const Tensor2<long long> &row_offset, const Tensor2<long long> &hash_key, size_t nnz,
    const Tensor2<float> &hash_table_value, Tensor2<size_t> &hash_value_index,
    Tensor2<__half> &embedding_feature, cudaStream_t stream);

}  // namespace HugeCTR
