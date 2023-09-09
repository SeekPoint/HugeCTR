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

#pragma once
#include <omp.h>

#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/embedding_data.hpp"
#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"

namespace HugeCTR {

template <typename TypeHashKey>
struct DistributedFilterKeyStorage {
  Tensor2<size_t> value_select_num;
  Tensor2<void> temp_value_select_storage;

  Tensor2<TypeHashKey> rowoffset_select;
  Tensor2<void> temp_rowoffset_select_scan_storage;

  DistributedFilterKeyStorage(const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf,
                              size_t max_nnz, size_t rowoffset_count, size_t global_id,
                              size_t global_num);
};

/**
 * The DistributedSlotSparseEmbeddingHash class inherits from Embedding class, which is the
 * base class for implementing all embedding layers. In this class, some of the slots in the
 * embedding table are assigned to multiple GPUs, which are called distributed slots. For
 * example, slot-0 on GPU-0/GPU-1, slot-1 on GPU-0/GPU-1, etc. The embedding table is encapsulated
 * in a hash table. The key in the hash table is called as hash_table_key, and the value in
 * the hash table is called as hash_table_value_index that means it indicates the embedding
 * feature's row number in the embedding table, and the embedding feature is called as
 * hash_table_value. This class implements all the operations needed by the training process of
 * embedding layer, including forward propagation and backward propagation. The forward propagation
 * is corresponding to the API forward(). The backward propagation is divided into 2-stage APIs:
 * backward() and update_params(). The class also provides the operations for uploading hash tables
 * (including hash_table_key, hash_table_value_index and hash_table_value) from a host file to
 * GPUs(which named load_parameters()), and for downloading hash tables from GPUs to
 * a host file(which named dump_parameters()).
 */
 /*
1.2 功能
在 DistributedSlotSparseEmbeddingHash 之中，嵌入表中的一些插槽被分配给多个GPU，称为分布式插槽。
例如，slot-0 被分配到GPU-0/GPU-1上，slot-1 被分配到GPU-0/GPU-1上。
嵌入表被封装在哈希表中，或者说哈希表是嵌入表的前置条件。哈希表一些相关成员变量如下：
      哈希表中的键称为 hash_table_key。
      哈希表中的值称为 hash_table_value_index，表示嵌入特征在嵌入表中的行号（row number）。
      嵌入特征称为 hash_table_value，就是利用 hash_table_value_index（行号）在嵌入表之中找到的那一行。

DistributedSlotSparseEmbeddingHash 类实现了嵌入层的训练过程所需的所有操作，包括前向传播和后向传播。
前向传播对应于API forward()。反向传播分为两个阶段的API：backward()和update_params()。
该类还提供将哈希表（包括哈希表键hash_table_key、hash_table_value_index和hash_table_value）从主机文件上载到GPU（load_parameters 方法）的操作，
以及将哈希表从GPU下载到主机文件（dump_parameters方法）的操作。

0x02 定义
    2.1 思路
    我们先自行想想看如何实现这个嵌入层，这样会让我们更好的理清楚思路。

    高维矩阵 ：假设不考虑field的情况下，embedding矩阵大小是 A * B，A 是 one-hot 的长度，B是embedding size，假如 one-hot 长10000000，embedding size是64。Hash_key 是一个one-hot [0,0,..0,1,0,..,0]，其可以定位到 embedding矩阵的一行。假设 hash_key 的第367位置上是1，则就会找到embedding矩阵的367行，从 367 行得到一个64长度的dense vector。
    数据特点 ：前面提到过，CTR的特点是高维，稀疏，这说明嵌入表10000000 行之中可能只有500行是有意义的数值，其余为空，
     **低维矩阵 ** ： HugeCTR 内部实际上不可能内部存放一个巨大矩阵，肯定是改用一个小型矩阵来存储，比如1000 x 64 的小型矩阵。
    转换机制 ：所以需要有一个机制，把 367 这个高维嵌入表的 row index 映射到这个小型低维矩阵的 row index，通过一系列复杂的操作用时间来换取空间。这也就是 DistributedSlotSparseEmbeddingHash 的一系列成员变量所起到的作用。
    2.2 代码
    DistributedSlotSparseEmbeddingHash 的定义如下，主要变量/概念为：
    CSR相关，可以结合CSR定义来印证。
           @param row_offset ：row_offset (CSR format of input sparse tensors)。
           @param hash_key ：value (CSR format of input sparse tensors)。
           @param nnz ：non-zero feature number per batch。
    输入/输出数据：
         embedding_data_ ：这里包括很多数据。
            前面提到的 DataReader.output_ 就会被保存在这里，就是 sparse input 信息。
            这里的 train_output_tensors_ 成员变量则是嵌入层最终的输出，就是多个GPU之间互相作用之后得出的输出。注意，train_output_tensors_ 在反向传播时候居然还被用来作为输入梯度。

    Hash相关：
            hash_tables_ ：这是一个哈希表vector，每一个元素都是一个hash_table（NvHashTable），本地每一个GPU对应这个vector之中的一个NvHashTable。目的是为了把高维矩阵的row offset 转换为低维矩阵的 row offset。
                  在 hash_table 内部，逻辑上来看每一个元素可以认为是 <key, value_index>（其实内部是个黑盒子，只是对外逻辑表示为一个哈希表 <key, value_index>）；
                  哈希表中的键称为 hash_table_key，其格式是 CSR (CSR format of input sparse tensors)相关。
                  哈希表中的值称为 hash_table_value_index，表示 CSR 对应的嵌入特征在嵌入表中的行号。
            hash_value_index_tensors_ ：embedding vector表的row index。就是低维矩阵的 row offset。
                  需要注意，其类型是 Tensors2，其类型是 std::vector<Tensor2>，所以每一个GPU对应了该vector之中的一个元素。
                  index 和 value 的行数相关。
                  内容是hash table value_index(row index of embedding)。
            hash_table_value_tensors_ ：embedding vector表的value。就是低维矩阵。
                  需要注意，其类型是 Tensors2，其类型是 std::vector<Tensor2>，所以每一个GPU对应了该vector之中的一个元素。
                  其内容是embedding vector。
                  用hash_value_index_tensors_的结果在这里查找一个 embedding vector。
    中间数据：
            embedding_feature_tensors_ ： 嵌入层前向传播的中间输出，就是上面查找到的embedding vector的结果，但是没有经过GPU之间的操作（ reduce-scatter等）；
            row_offset_allreduce_tensors_ ：allreduce之后的row_offset。
    反向传播：
          wgrad_tensors_ ：后向传播的梯度，是backward之后产生的结果；
          embedding_optimizers_ : 嵌入层对应的优化器。
    这里有两点说明：
          为了方便起见，hash_value_index_tensors_ 这样虽然是一个向量的向量，我们后续都省略一步，当作向量来考虑。
          需要对 hash_value_index_tensors_ 做进一步解释：
                 hash_value_index_tensors_ 起到了解耦合的作用，把低维矩阵和哈希表进行解耦合。因为解耦合的原因，hash_value_index_tensors_ 并不应该知道 哈希表内部把高维矩阵的维度映射到了多大的低维矩阵，而 hash_value_index_tensors_ 大小也不应该随之变化。
                 所以，hash_value_index_tensors_ 大小被固定为：batch_size * nnz_per_slot，可以认为就是CSR之中元素个数。因此 hash_value_index_tensors_ 实际上记录了每个元素对应的低维矩阵offset 数值，hash_value_index_tensors_ 其事实上就是和CSR之中元素位置一一对应。
                 因此，最终嵌入表查找时候，是通过CSR row offset 来找到 CSR之中每个元素，也找到了hash_value_index_tensors_ 这个表的index，从而就能找到其低维矩阵offset。
我们再从源码之中找出部分注释给大家看看几个变量之间的关系，其查找逻辑是从上到下。
005-001.jpg
DistributedSlotSparseEmbeddingHash 具体定义如下：
*/
template <typename TypeHashKey, typename TypeEmbeddingComp>
class DistributedSlotSparseEmbeddingHash : public IEmbedding {
  using NvHashTable = HashTable<TypeHashKey, size_t>;

 private:
  // 前面提到的 DataReader.output_ 就会被保存在这里。就是sparse input信息
  EmbeddingData<TypeHashKey, TypeEmbeddingComp> embedding_data_;
/*
 * 3.1.3 临时存储
  前面cub方法之中，都要有一个临时存储区域，
  因此 DistributedSlotSparseEmbeddingHash 之中有一个 DistributedFilterKeyStorage 就是用来达到这个目的。
  */
  // 是 hash_value, hash_value_index的实际存储位置
  std::vector<DistributedFilterKeyStorage<TypeHashKey>> filter_keys_storage_;

  std::vector<std::shared_ptr<NvHashTable>> hash_tables_; /**< Hash table.  */

  // define tensors
  Tensors2<float> hash_table_value_tensors_;  /**< Hash table value. */
  Tensors2<size_t> hash_value_index_tensors_; /**< Hash table value index. The index is
                                                   corresponding to the line number of the value. */
  Tensors2<TypeEmbeddingComp>
      embedding_feature_tensors_;             /**< the output tensor of the forward(). */
  Tensors2<TypeEmbeddingComp> wgrad_tensors_; /**< the input tensor of the backward(). */

  Tensors2<TypeHashKey>
      row_offset_allreduce_tensors_; /**< The temp memory to store the row_offset after all_reduce
                                        operation among multi-gpu in forward(). */

  Tensors2<TypeEmbeddingComp> utest_forward_temp_tensors_;

  size_t max_vocabulary_size_;         /**< Max vocabulary size for each GPU. */
  size_t max_vocabulary_size_per_gpu_; /**< Max vocabulary size for each GPU. */

  SparseEmbeddingFunctors functors_;

  std::vector<EmbeddingOptimizer<TypeHashKey, TypeEmbeddingComp>> embedding_optimizers_;

  /**
   * Initialize the embedding table on local GPUs.
   * @param max_vocabulary_size_per_gpu max vocabulary size per GPU.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors embedding table tensors.
   * @param resource_manager GPU device resources.
   */
  void init_embedding(size_t max_vocabulary_size_per_gpu, size_t embedding_vec_size,
                      Tensors2<float> &hash_table_value_tensors);

  /**
   * load_parameters for DistributedSlotSparseEmbeddingHash.
   * @param keys the memory buffer storing keys.
   * @param embeddings the memory buffer storing embedding vectors.
   * @param num the number of unique keys (embedding vectors) in keys (embeddings).
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param max_vocabulary_size_per_gpu max vocabulary size for each GPU
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void load_parameters(const Tensor2<TypeHashKey> &keys, const Tensor2<float> &embeddings,
                       size_t num, size_t vocabulary_size, size_t embedding_vec_size,
                       size_t max_vocabulary_size_per_gpu, Tensors2<float> &embedding_tensors,
                       std::vector<std::shared_ptr<NvHashTable>> &hash_tables);

  void load_parameters(BufferBag &buf_bag, size_t num, size_t vocabulary_size,
                       size_t embedding_vec_size, size_t max_vocabulary_size_per_gpu,
                       Tensors2<float> &embedding_tensors,
                       std::vector<std::shared_ptr<NvHashTable>> &hash_tables);

  /**
   * dump_parameters for DistributedSlotSparseEmbeddingHash
   * download hash_table from GPUs to CPU.
   * @param sparse_model the folder name of sparse model.
   * @param vocabulary_size the total row number of hash table.
   * @param embedding_vec_size embedding vector size.
   * @param hash_table_value_tensors the tensors of hash table value on multi GPUs.
   * @param hash_tables the hash tables on multi GPUs
   * @param device_resources all gpus device resources.
   * @param context gpu device context, for switching device
   */
  void dump_parameters(
      const std::string &sparse_model, size_t vocabulary_size, size_t embedding_vec_size,
      const Tensors2<float> &hash_table_value_tensors,
      const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const;
  void dump_parameters(
      Tensor2<TypeHashKey> &keys, Tensor2<float> &embeddings, size_t *num, size_t vocabulary_size,
      size_t embedding_vec_size, const Tensors2<float> &embedding_tensors,
      const std::vector<std::shared_ptr<HashTable<TypeHashKey, size_t>>> &hash_tables) const;

 public:
  /**
   * The constructor of DistributedSlotSparseEmbeddingHash.
   * @param row_offsets_tensors row offsets of the input tensor(refer to row offset vector in sparse
   * matrix CSR format).
   * @param hash_key_tensors hash keys of the input tensor(refer to value vector in sparse matrix
   * CSR format).
   * @param embedding_params embedding params for initialization.
   * @param resource_manager the GPU resource group
   */
  DistributedSlotSparseEmbeddingHash(const Tensors2<TypeHashKey> &train_row_offsets_tensors,
                                     const Tensors2<TypeHashKey> &train_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &train_nnz_array,
                                     const Tensors2<TypeHashKey> &evaluate_row_offsets_tensors,
                                     const Tensors2<TypeHashKey> &evaluate_value_tensors,
                                     const std::vector<std::shared_ptr<size_t>> &evaluate_nnz_array,
                                     const SparseEmbeddingHashParams &embedding_params,
                                     const std::shared_ptr<ResourceManager> &resource_manager);

  DistributedSlotSparseEmbeddingHash(const SparseTensors<TypeHashKey> &train_keys,
                                     const SparseTensors<TypeHashKey> &evaluate_keys,
                                     const SparseEmbeddingHashParams &embedding_params,
                                     const std::shared_ptr<ResourceManager> &resource_manager);

  void filter_keys_per_gpu(bool is_train, size_t id, size_t global_id, size_t global_num);

  /**
   * The forward propagation of embedding layer.
5.3 怎么得到row_offset
5.3.1 问题
 目前，我们只设置了EmbeddingData的train_keys/train_value_tensors_，但这是SparseTensor，其内部不仅仅有value，还有row_offset等专门针对稀疏矩阵的信息，所以这部分也要进行设置。

我们提前看看前向传播，会发现其使用了类似 embedding_data_.get_row_offsets_tensors 进行运算。
   但是我们目前并没有配置这样的参数，只是配置了 train_keys。
   这个地方很绕，仔细看代码，原来在前向传播之中有使用 filter_keys_per_gpu 进行设置类似参数。


/*
0x02 总体逻辑
前向传播的总体功能是：Embedded_lookuper负责本地gpu计算和查找嵌入向量，即用户输入->嵌入向量。这里只考虑 *train* 名字的各种变量，忽略 *evalute* 名字的各种变量，即只看训练逻辑。
2.1 注释&思路
码之中的注释如下：
      ...
我们翻译梳理逻辑如下 Read data from input_buffers_ -> look up -> write to output_tensors，具体就是：
      从input_buffers_读取数据。具体是通过 filter_keys_per_gpu 来完成对 embedding_data_ 的一系列配置。
      从embedding之中进行 look up，即调用 functors_.forward_per_gpu 从本gpu的hashmap做lookup操作。
            由 DistributedSlotSparseEmbeddingHash 的特点我们知道，因为当前gpu对应的数据key都在此gpu，所以此时不需要做节点间通信。
            这里 hash_tables_[i]，hash_table_value_tensors_[i]，hash_value_index_tensors_[i] 就是本地第 i 个GPU对应的hashmap组合。
            embedding_data_.get_value_tensors(is_train)[i] 就是从我们之前提到的GPU sparse input 内部提取的输入训练数据。
            进行本地规约。
      做reduce scatter操作。每个gpu的数据是batch size条，但是每条数据里的每个slot只是一部分key，需要做reduce scatter操作，做完reduce scatter后，数据才是完整的，此时每个gpu上分到完整数据的一部分。
      写到output_tensors。
具体一些成员变量的定义需要回忆一下。
      hash_value_index_tensors_ ：embedding vector表的row index。就是低维矩阵的 row offset。
          需要注意，其类型是 Tensors2，其类型是 std::vector<Tensor2>，所以每一个GPU对应该vector之中的一个元素。
          index 和 value 的行数相关。
          内容是hash table value_index(row index of embedding)。
      hash_table_value_tensors_ ：embedding vector表的value。就是低维矩阵。
          需要注意，其类型是 Tensors2，其类型是 std::vector<Tensor2>，所以每一个GPU对应该vector之中的一个元素。
          其内容是embedding vector。
          用hash_value_index_tensors_的结果在这里查找一个 embedding vector。
后续我们依然做简化，忽略多个 worker，多个 GPU 的情况。
2.2 总体代码
      前向传播总体代码如下：
      本地多个GPU并行前向传播，每个线程对应一个GPU，多GPU进行。
      调用 filter_keys_per_gpu 完成完成了对 EmbeddingData 的配置，这里i就是GPU index，拿到本GPU对应的输入数据。
      调用 forward_per_gpu 从本gpu的hashmap做lookup操作。
      reduce scatter，做了之后，数据才是完整的，每个gpu上分到完整数据的一部分。
      all_reduce 操作，这是combiner=mean时需要继续处理。
      forward_scale 操作，做平均。

具体流程是
006-001.jpg
*/
*/
  void forward(bool is_train, int eval_batch = -1) override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    {
      // 本地多个GPU并行前向传播
      // 每个线程对应一个GPU，多GPU进行
      size_t i = omp_get_thread_num();  // 拿到本线程序号
      CudaDeviceContext context(embedding_data_.get_local_gpu(i).get_device_id());
      if (embedding_data_.embedding_params_.is_data_parallel) {
        // 这里完成了对 EmbeddingData 的配置，这里i就是GPU index
        // 在这里有操作
        filter_keys_per_gpu(is_train, i, embedding_data_.get_local_gpu(i).get_global_id(),
                            embedding_data_.get_resource_manager().get_global_gpu_count());
      }
      // 从本gpu的hashmap做lookup操作
      // 这里 hash_tables_[i]，hash_table_value_tensors_[i]，hash_value_index_tensors_[i] 就是对应的hashmap
      // 部分前向操作
      /*
 0x04 Lookup操作
此部分就是完成嵌入表 look up操作。现在EmbeddingData得到了各种配置，就是sparse input参数，
所以可以利用其作为key，得到embedding vector了。这部分是在 forward_per_gpu 内部完成的
       */
      functors_.forward_per_gpu(embedding_data_.embedding_params_.get_batch_size(is_train),
                                embedding_data_.embedding_params_.slot_num,
                                embedding_data_.embedding_params_.embedding_vec_size, 0, is_train,
                                embedding_data_.get_row_offsets_tensors(is_train)[i],
                                embedding_data_.get_value_tensors(is_train)[i],
                                *embedding_data_.get_nnz_array(is_train)[i], *hash_tables_[i],
                                hash_table_value_tensors_[i], hash_value_index_tensors_[i],
                                embedding_feature_tensors_[i],
                                embedding_data_.get_local_gpu(i).get_stream());
    }

    /*
0x05 Reduce Scatter
现在每个GPU之上都得到了自己样本对应的稠密向量，记录在 embedding_feature_tensors_ 之上。
每个GPU的数据是 batch size 条，每条有 slot number 个稠密向量，我们现在回忆一下：
DistributedSlotEmbeddingHash：
所有特征都存储于不同特征域/槽上，不管槽索引号是多少，这些特征都根据特征的索引号分布到不同的GPU上。
这意味着同一插槽中的特征可能存储在不同的 GPU 中，这就是将其称为“分布式插槽”的原因。由于需要全局规约，
所以DistributedSlotEmbedding 适合 embedding 大于 GPU 内存大小的情况，
因而DistributedSlotEmbedding 在 GPU 之间有更多的内存交换。

GPU之上每个样本数据之中的slot只是slot的一部分数据，我们给出一个例子。
我们假设一共有2个gpu，batch size为2，一共3个slot。
有两个样本，拿第一个样本为例，slot 1有两个key，分别是GPU 1 上的1，GPU 2上的7。
所以需要把这两个key进行归并操作。具体如下：
006-008.png
每条数据里面的每个slot都只是一部分key，同一插槽中的特征可能存储在不同的 GPU 中，这些特征都根据特征的索引号分布到不同的GPU上。
这样就需要把GPU 1，GPU 2之上的数据进行合并，做完reduce scatter后，数据应该是完整的，并且每个gpu上只分到一部分完整的数据。
5.1 背景知识
关于 Reduce Scatter 的原理，请参见 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html。这里只是大致介绍。
Reduce操作的作用是对所有计算节点上的数值进行归约操作，并且只将归约后的结果保存到主节点上。和 AllReduce 的操作其实一样，只不过是把结果只放到root上而已。
006-009.png
ReduceScatter 操作执行与Reduce操作相同的操作，只是结果分散在rank之间的相等块中，
每个rank根据其rank索引获得一块数据，即每个rank只受到reduce结果的一部分数据。
或者说，ReduceScatter 就是先做Scatter，将数据切分成同等大小的数据块，再依据Rank Index 对每一个rank所获得的数据做Reduce。
这类似于全聚集，但是并不是将数据简单拼接到一起而是做了规约操作（比如，求和或最大值操作）。
006-010.png
或者参见下图，来自NVIDIA文档 https://images.nvidia.cn/events/sc15/pdfs/NCCL-Woolley.pdf。
对所有GPU上的数据进行reduce操作，这里是sum，然后将结果切分到所有的GPU上。
006-011.png
 5.2 代码
具体代码如下，是对 embedding_feature_tensors_ 进行 reduce scatter，结果放在 embedding_data_.get_output_tensors(is_train) 之上
*/
    // do reduce scatter
    // do reduce scatter
    // 做了之后，数据才是完整的，每个gpu上分到完整数据的一部分
    size_t recv_count = embedding_data_.get_batch_size_per_gpu(is_train) *
                        embedding_data_.embedding_params_.slot_num *
                        embedding_data_.embedding_params_.embedding_vec_size;
    functors_.reduce_scatter(recv_count, embedding_feature_tensors_,
                             embedding_data_.get_output_tensors(is_train),
                             embedding_data_.get_resource_manager());

    // scale for combiner=mean after reduction
    if (embedding_data_.embedding_params_.combiner == 1) {
      size_t send_count = embedding_data_.embedding_params_.get_batch_size(is_train) *
                              embedding_data_.embedding_params_.slot_num +
                          1;
      functors_.all_reduce(send_count, embedding_data_.get_row_offsets_tensors(is_train),
                           row_offset_allreduce_tensors_, embedding_data_.get_resource_manager());

      // do average
      /*6.2 Forward Scale
最后要做一步 Forward Scale 操作。
前面我们做了AllReduce之后，得到 row_offset_allreduce_tensors_ 是 0,9,14,19,21。
这样就知道第一个行总个数是9个，第二行总个是是7+7-9 = 5个。
就可以对embedding_data_.get_output_tensors(is_train)的每个元素进行操作，每个元素都除以本slot的元素总数，就是做mean了。
       */
      functors_.forward_scale(
          embedding_data_.embedding_params_.get_batch_size(is_train),
          embedding_data_.embedding_params_.slot_num,
          embedding_data_.embedding_params_.embedding_vec_size, row_offset_allreduce_tensors_,
          embedding_data_.get_output_tensors(is_train), embedding_data_.get_resource_manager());
    }

    return;
  }
/*
  0x04 backward
  4.1 总体代码
  由之前分析我们可以知道，反向传播时候，输入的梯度就是存储在embedding_data_.get_output_tensors(true)之中。
  总体代码分为两部分，第一步是使用all-gather 操作来在每个GPU之上都收集到所有样本的全部梯度。
  第二步是调用 functors_.backward进行计算。
 */
  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
   */
  void backward() override {
    // Read dgrad from output_tensors -> compute wgrad

    // do all-gather to collect the top_grad
    size_t send_count = embedding_data_.get_batch_size_per_gpu(true) *
                        embedding_data_.embedding_params_.slot_num *
                        embedding_data_.embedding_params_.embedding_vec_size;
    functors_.all_gather(send_count, embedding_data_.get_output_tensors(true),
                         embedding_feature_tensors_, embedding_data_.get_resource_manager());

    // do backward
    /*
4.3 backward
这部分完成如下功能：计算本地每个gpu上的梯度。此函数完成之后，wgrad_tensors_ 成员变量就是本GPU计算产生的新梯度。
calculating wgrad，会选择如下两种之一：
    sum: calling backward_sum_kernel() ；
    mean: calling backward_mean_kernel()；
     */
    functors_.backward(embedding_data_.embedding_params_.get_batch_size(true),
                       embedding_data_.embedding_params_.slot_num,
                       embedding_data_.embedding_params_.embedding_vec_size,
                       embedding_data_.embedding_params_.combiner, row_offset_allreduce_tensors_,
                       embedding_feature_tensors_, wgrad_tensors_,
                       embedding_data_.get_resource_manager());

    return;
  }

  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the hash table by wgrad(from backward()) and optimizer.
   */
  void update_params() override {
    // accumulate times for adam optimizer
    embedding_data_.embedding_params_.opt_params.hyperparams.adam.times++;
#pragma omp parallel num_threads(embedding_data_.get_resource_manager().get_local_gpu_count())
    {
      size_t id = omp_get_thread_num();
      CudaDeviceContext context(embedding_data_.get_local_gpu(id).get_device_id());
      // do update params operation
      embedding_optimizers_[id].update(
          embedding_data_.embedding_params_.get_batch_size(true),
          embedding_data_.embedding_params_.slot_num,
          embedding_data_.embedding_params_.embedding_vec_size, max_vocabulary_size_per_gpu_,
          *embedding_data_.get_nnz_array(true)[id],
          embedding_data_.get_row_offsets_tensors(true)[id], hash_value_index_tensors_[id],
          wgrad_tensors_[id], hash_table_value_tensors_[id],
          embedding_data_.get_local_gpu(id).get_sm_count(),
          embedding_data_.get_local_gpu(id).get_stream());
    }

    return;
  }

  /**
   * Initialize the embedding table
   */
  void init_params() override {
    init_embedding(max_vocabulary_size_per_gpu_,
                   embedding_data_.embedding_params_.embedding_vec_size, hash_table_value_tensors_);
  }

  /**
   * Read the hash table from the weight_stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param sparse_model the folder name of sparse model.
   */
  void load_parameters(std::string sparse_model) override;
  void load_parameters(BufferBag &buf_bag, size_t num) override;

  /**
   * Download the hash table from multi-GPUs global memroy to CPU memory
   * and write it to the weight_stream on the host.
   * @param sparse_model the folder name of sparse model.
   */
  void dump_parameters(std::string sparse_model) const override;
  void dump_parameters(BufferBag &buf_bag, size_t *num) const override;

  void dump_opt_states(std::ofstream &stream) override;
  void load_opt_states(std::ifstream &stream) override;
  void reset_optimizer() override;

  /**
   * Reset the embedding
   */
  void reset() override;

  /**
   * Get the total size of hash tables on all GPUs.
   */
  size_t get_params_num() const override {
    // Read data from input_buffers_ -> look up -> write to output_tensors

    return get_vocabulary_size() * embedding_data_.embedding_params_.embedding_vec_size;
  }

  size_t get_vocabulary_size() const override {
    size_t total_size = 0;

    // need to collect the <key, value> pair count from all GPUs and do sum reduction
    CudaDeviceContext context;
    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
      total_size += hash_tables_[id]->get_size(embedding_data_.get_local_gpu(id).get_stream());
    }

    return total_size;
  }

  size_t get_max_vocabulary_size() const override { return max_vocabulary_size_; }

  /*
4.5 输出矩阵
我们这里通过一个函数来看输出稠密矩阵的大小，其就是 batch_size_per_gpu * slot_num * embedding_vec_size。
   * */
  // only used for results check
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  void get_forward_results(bool is_train, Tensor2<TypeEmbeddingComp> &embedding_feature) {
    size_t memcpy_size = embedding_data_.get_batch_size_per_gpu(is_train) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;

    functors_.get_forward_results(memcpy_size, embedding_data_.get_output_tensors(is_train),
                                  embedding_feature, utest_forward_temp_tensors_,
                                  embedding_data_.get_resource_manager());

    return;
  }

  /**
   * Get the forward() results from GPUs and copy them to TensorFlow's tensor.
   */
  void get_forward_results_tf(const bool is_train, const bool on_gpu,
                              void *const forward_result) override {
    size_t memcpy_size = embedding_data_.get_batch_size_per_gpu(is_train) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;
    functors_.get_forward_results(memcpy_size, embedding_data_.get_output_tensors(is_train),
                                  forward_result, utest_forward_temp_tensors_,
                                  embedding_data_.get_resource_manager(), on_gpu);
    return;
  }

  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  void get_backward_results(Tensor2<TypeEmbeddingComp> &wgrad, int devIndex) {
    // wgard shuld be the same on multi-gpus after backward()
    size_t memcpy_size = embedding_data_.embedding_params_.get_batch_size(true) *
                         embedding_data_.embedding_params_.slot_num *
                         embedding_data_.embedding_params_.embedding_vec_size;

    functors_.get_backward_results(devIndex, memcpy_size, wgrad_tensors_, wgrad,
                                   embedding_data_.get_resource_manager());

    return;
  }

  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  void get_update_params_results(Tensor2<TypeHashKey> &hash_table_key,
                                 Tensor2<float> &hash_table_value) {
    functors_.get_update_params_results(embedding_data_.embedding_params_.embedding_vec_size,
                                        max_vocabulary_size_, hash_table_value_tensors_,
                                        hash_tables_, hash_table_key, hash_table_value,
                                        embedding_data_.get_resource_manager());

    return;
  }

  /**
   * Check overflow
   * @param lr
   */
  void check_overflow() const override {
    CudaDeviceContext context;

    for (size_t id = 0; id < embedding_data_.get_resource_manager().get_local_gpu_count(); id++) {
      context.set_device(embedding_data_.get_local_gpu(id).get_device_id());
      size_t count = hash_tables_[id]->get_size(embedding_data_.get_local_gpu(id).get_stream());
      if (count > max_vocabulary_size_per_gpu_) {
        CK_THROW_(Error_t::OutOfBound,
                  "Runtime vocabulary size (" + std::to_string(count) +
                      ") exceeds max_vocabulary_size_per_gpu (" +
                      std::to_string(max_vocabulary_size_per_gpu_) + ") on GPU " +
                      std::to_string(embedding_data_.get_local_gpu(id).get_device_id()) +
                      ", new feature insertion failed.\n");
      }
    }
  }

  /** only used in tf embedding plugin to distribute top_gradients to each GPUs' output tensor.
   */
  cudaError_t update_top_gradients(const bool on_gpu, const void *const top_gradients) override {
    auto output_tensors = embedding_data_.get_output_tensors(true);
    CudaDeviceContext context;

    const auto top_gradients_internel = reinterpret_cast<const TypeEmbeddingComp *>(top_gradients);
    cudaMemcpyKind direction = (on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);

    cudaError_t error = cudaError_t::cudaSuccess;
    for (size_t dev_id = 0; dev_id < embedding_data_.get_resource_manager().get_local_gpu_count();
         ++dev_id) {
      context.set_device(embedding_data_.get_local_gpu(dev_id).get_device_id());

      error = cudaMemcpyAsync(
          output_tensors[dev_id].get_ptr(),
          top_gradients_internel + dev_id * output_tensors[dev_id].get_num_elements(),
          output_tensors[dev_id].get_size_in_bytes(), direction,
          embedding_data_.get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    for (size_t dev_id = 0; dev_id < embedding_data_.get_resource_manager().get_local_gpu_count();
         ++dev_id) {
      context.set_device(embedding_data_.get_local_gpu(dev_id).get_device_id());
      error = cudaStreamSynchronize(embedding_data_.get_local_gpu(dev_id).get_stream());
      if (error != cudaError_t::cudaSuccess) return error;
    }

    return cudaError_t::cudaSuccess;
  }

  USE_EMBEDDING_DATA_FUNCTION(embedding_data_)
};  // end of class DistributedSlotSparseEmbeddingHash

}  // namespace HugeCTR
