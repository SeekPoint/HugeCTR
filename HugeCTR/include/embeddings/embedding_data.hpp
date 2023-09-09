/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <embedding.hpp>
#include <general_buffer2.hpp>
#include <resource_manager.hpp>
#include <unordered_map>

#include "HugeCTR/include/embeddings/sparse_embedding_functors.hpp"
#include "HugeCTR/include/utils.hpp"
namespace HugeCTR {
/*
0x05 EmbeddingData
前面提到了 DistributedSlotSparseEmbeddingHash 如下成员变量会保存一些嵌入表信息。
EmbeddingData<TypeHashKey, TypeEmbeddingComp> embedding_data_;
我们来挖掘一下。
5.1 定义
        EmbeddingData 定义如下，这里有两套成员变量，Tensors2 和 SparseTensors。
        Tensors2 如下：
                train_value_tensors_ 这个就会记录sparse input，是CSR 的value。
                train_row_offsets_tensors_ 是CSR 的 row offset。
                train_nnz_array_ 是CSR 相关的nnz。
                train_output_tensors_ 这个是前向传播的输出。
        SparseTensors 如下：
                train_keys_ 会把 value，offset，nnz都整合在一起，这里怀疑是在接口迁移，所以维护了两套。为何迁移？
                因为train_value_tensors_，train_row_offsets_tensors_，train_nnz_array_ 都是Tensor2，是普通张量，而 train_keys_ 是 SparseTensors，可以一个变量就搞定前面所有概念。
                valuate_keys_ 是验证集相关。
所以，embedding_data_ 就是包揽了嵌入层的输入和输出。
 需要注意的是，这里都是 Tensors2，可以认为是 Tensor2 的列表，列表之中每一个Tensor2 对应了一个GPU。
 *
 *
 *
 * 0x03 配置数据
之前，在EmbeddingData 初始化时候，只是配置了其成员函数 train_keys_，
 train_keys_ 就是前面提到的 sparse_input，就是CSR format对应的稀疏张量。

 此时数据如下, embedding_offsets_ 和 train_output_tensors_ 都是预先分配的，
 我们假设 CSR 数据为 ：40,50,10,20,30,50,10,30,20，CSR row offset 是 0,4,7,9。
 006-002.jpg
 */
template <typename TypeKey, typename TypeEmbeddingComp>
class EmbeddingData {
 public:
  const Embedding_t embedding_type_;
  SparseEmbeddingHashParams embedding_params_; /**< Sparse embedding hash params. */

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
      bufs_;                                         /**< The buffer for storing output tensors. */
  Tensors2<TypeEmbeddingComp> train_output_tensors_; /**< The output tensors. */
  Tensors2<TypeEmbeddingComp> evaluate_output_tensors_; /**< The output tensors. */
  Tensors2<TypeKey> train_row_offsets_tensors_; /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> train_value_tensors_;       /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> train_nnz_array_;
  Tensors2<TypeKey>
      evaluate_row_offsets_tensors_;         /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> evaluate_value_tensors_; /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> evaluate_nnz_array_;

  std::shared_ptr<ResourceManager> resource_manager_; /**< The GPU device resources. */

  SparseTensors<TypeKey> train_keys_;
  SparseTensors<TypeKey> evaluate_keys_;
  Tensors2<TypeKey> embedding_offsets_;

  size_t get_batch_size_per_gpu(bool is_train) const {
    return embedding_params_.get_batch_size(is_train) / resource_manager_->get_global_gpu_count();
  }

  size_t get_universal_batch_size_per_gpu() const {
    return embedding_params_.get_universal_batch_size() / resource_manager_->get_global_gpu_count();
  }

  ResourceManager& get_resource_manager() const { return *resource_manager_; }

  const GPUResource& get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& get_buffer(size_t i) const {
    return bufs_[i];
  }

  Tensors2<TypeEmbeddingComp>& get_output_tensors(bool is_train) {
    if (is_train) {
      return train_output_tensors_;
    } else {
      return evaluate_output_tensors_;
    }
  }

  SparseTensors<TypeKey>& get_input_keys(bool is_train) {
    return is_train ? train_keys_ : evaluate_keys_;
  }
/*
   5.3.2 引用
 我们仔细看看 EmbeddingData 的一些成员函数，发现他们都返回了引用。这就是关键，这些成员函数可以修改 EmbeddingData的内部成员变量，比如：get_row_offsets_tensors返回了一个引用。
 类似的，比如get_output_tensors，get_input_keys，get_row_offsets_tensors，get_value_tensors，get_nnz_array 都返回引用，这说明 EmbeddingData 大部分成员变量都是可以被引用来修改的。

   从 CSR 读取 offset 代码如下：
     因为输入有几千万个，但是可能其中只有几百个才非零，所以hash表就是把这几千万个输入做第一次映射，可以减少大量内存空间。
 * */
  Tensors2<TypeKey>& get_row_offsets_tensors(bool is_train) {
    if (is_train) {
      return train_row_offsets_tensors_;
    } else {
      return evaluate_row_offsets_tensors_;
    }
  }
/*4.1 提取数据
  这里用到了比如 get_row_offsets_tensors 这样的方法从 embedding_data_ 之中提取输入数据。
  从input_buffers_读取数据对应的提取数据代码如下，就是从GPU的sparse input csr数据中读取到输入数据，
  作为后续在hash table查找的key
  */
  Tensors2<TypeKey>& get_value_tensors(bool is_train) {
    if (is_train) {
      return train_value_tensors_;
    } else {
      return evaluate_value_tensors_;
    }
  }

  std::vector<std::shared_ptr<size_t>>& get_nnz_array(bool is_train) {
    if (is_train) {
      return train_nnz_array_;
    } else {
      return evaluate_nnz_array_;
    }
  }

  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param resource_manager the GPU device resource group
   * @param scaler scaler factor for mixed precision
5.2 构建
这里有两套构建函数，可能维护者在从旧接口切换到新接口。
   结合前后文，sparse_input 在 DistributedSlotSparseEmbeddingHash 构造函数之中是 train_keys 参数，
   在EmbeddingData 这里就是train_value_tensors，
   所以，value_tensors 就是我们要关注的，
   从注释可以知道，这是输入数据的value tensors，指向了稀疏矩阵的 value vector。

我们最终拓展如下，经过第 C 步之后，DistributedSlotSparseEmbeddingHash的成员变量 也指向了 GPU 内存，
   这里依据构建函数的不同，train_output_tensors_，和 train_keys_ 可能（可能是因为有两种不同的构造方式，目前只是讨论其中一种）都会指向用户输入训练数据。
005-003.jpg

  */
  EmbeddingData(const Tensors2<TypeKey>& train_row_offsets_tensors,
                const Tensors2<TypeKey>& train_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
                const Tensors2<TypeKey>& evaluate_row_offsets_tensors,
                const Tensors2<TypeKey>& evaluate_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& evaluate_nnz_array,
                const Embedding_t embedding_type, const SparseEmbeddingHashParams& embedding_params,
                const std::shared_ptr<ResourceManager>& resource_manager)
      : embedding_type_(embedding_type),
        embedding_params_(embedding_params),
        train_row_offsets_tensors_(train_row_offsets_tensors),
        train_value_tensors_(train_value_tensors),
        train_nnz_array_(train_nnz_array),
        evaluate_row_offsets_tensors_(evaluate_row_offsets_tensors),
        evaluate_value_tensors_(evaluate_value_tensors),
        evaluate_nnz_array_(evaluate_nnz_array),
        resource_manager_(resource_manager) {
    try {
      // Error check
      if (embedding_params.train_batch_size < 1 || embedding_params.evaluate_batch_size < 1 ||
          embedding_params.slot_num < 1 || embedding_params.embedding_vec_size < 1) {
        CK_THROW_(Error_t::WrongInput, "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
      }

      if (embedding_params.embedding_vec_size > 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "the embedding_vec_size can not be more than 1024 in embedding layer");
      }

      size_t total_gpu_count = resource_manager_->get_global_gpu_count();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();

      if (train_row_offsets_tensors.size() != local_gpu_count ||
          train_value_tensors.size() != local_gpu_count ||
          evaluate_row_offsets_tensors.size() != local_gpu_count ||
          evaluate_value_tensors.size() != local_gpu_count) {
        CK_THROW_(
            Error_t::WrongInput,
            "either row_offsets_tensors.size() or value_tensors.size() isn't local_gpu_count_");
      }

      assert(bufs_.empty());
      for (size_t i = 0; i < local_gpu_count; i++) {
        std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf =
            GeneralBuffer2<CudaAllocator>::create();
        bufs_.push_back(buf);

        Tensor2<TypeEmbeddingComp> tensor;
        buf->reserve({get_batch_size_per_gpu(true), embedding_params_.slot_num,
                      embedding_params_.embedding_vec_size},
                     &tensor);
        train_output_tensors_.push_back(tensor);
        buf->reserve({get_batch_size_per_gpu(false), embedding_params_.slot_num,
                      embedding_params_.embedding_vec_size},
                     &tensor);
        evaluate_output_tensors_.push_back(tensor);
      }

      // value，offset，nnz又整合了进来
      for (size_t i = 0; i < local_gpu_count; i++) {
        train_keys_.emplace_back(train_value_tensors_[i], train_row_offsets_tensors_[i],
                                 train_nnz_array_[i]);
        evaluate_keys_.emplace_back(evaluate_value_tensors_[i], evaluate_row_offsets_tensors_[i],
                                    evaluate_nnz_array_[i]);
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
    return;
  }

  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param resource_manager the GPU device resource group
   * @param scaler scaler factor for mixed precision
   */
  EmbeddingData(const Embedding_t embedding_type, const SparseTensors<TypeKey>& train_keys,
                const SparseTensors<TypeKey>& evaluate_keys,
                const SparseEmbeddingHashParams& embedding_params,
                const std::shared_ptr<ResourceManager>& resource_manager)
      : embedding_type_(embedding_type),
        embedding_params_(embedding_params),
        train_keys_(train_keys),
        evaluate_keys_(evaluate_keys),
        resource_manager_(resource_manager) {
    try {
      // Error check
      if (embedding_params.train_batch_size < 1 || embedding_params.evaluate_batch_size < 1 ||
          embedding_params.slot_num < 1 || embedding_params.embedding_vec_size < 1) {
        CK_THROW_(Error_t::WrongInput, "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
      }

      if (embedding_params.embedding_vec_size > 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "the embedding_vec_size can not be more than 1024 in embedding layer");
      }

      size_t total_gpu_count = resource_manager_->get_global_gpu_count();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();

      assert(bufs_.empty());
      for (size_t i = 0; i < local_gpu_count; i++) {
        CudaDeviceContext context(get_local_gpu(i).get_device_id());
        auto buf = GeneralBuffer2<CudaAllocator>::create();
        bufs_.push_back(buf);

        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({get_batch_size_per_gpu(true), embedding_params_.slot_num,
                        embedding_params_.embedding_vec_size},
                       &tensor);
          train_output_tensors_.push_back(tensor);
        }
        {
          Tensor2<TypeEmbeddingComp> tensor;
          buf->reserve({get_batch_size_per_gpu(false), embedding_params_.slot_num,
                        embedding_params_.embedding_vec_size},
                       &tensor);
          evaluate_output_tensors_.push_back(tensor);
        }
        {
          Tensor2<TypeKey> tensor;
          buf->reserve({embedding_params.slot_size_array.size()}, &tensor);
          embedding_offsets_.push_back(tensor);
        }
      }

    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
    return;
  }
  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  DISALLOW_COPY_AND_MOVE(EmbeddingData)
};
//输出在哪里？就是在 embedding_data.train_output_tensors_ 之中，后续我们会分析。
//所以，对于 embedding，就是通过sparse_input_map 和 train_tensor_entries_list 构成了输入，输出数据流。
#define USE_EMBEDDING_DATA_FUNCTION(embedding_data)                                          \
  Embedding_t get_embedding_type() const override { return embedding_data.embedding_type_; } \
  std::vector<TensorBag2> get_train_output_tensors() const override {                        \
    std::vector<TensorBag2> bags;                                                            \
    for (const auto& t : embedding_data.train_output_tensors_) {                             \
      bags.push_back(t.shrink());                                                            \
    }                                                                                        \
    return bags;                                                                             \
  }                                                                                          \
  std::vector<TensorBag2> get_evaluate_output_tensors() const override {                     \
    std::vector<TensorBag2> bags;                                                            \
    for (const auto& t : embedding_data.evaluate_output_tensors_) {                          \
      bags.push_back(t.shrink());                                                            \
    }                                                                                        \
    return bags;                                                                             \
  }                                                                                          \
  void set_learning_rate(float lr) override {                                                \
    embedding_data.embedding_params_.opt_params.lr = lr;                                     \
  }                                                                                          \
  const SparseEmbeddingHashParams& get_embedding_params() const override {                   \
    return embedding_data.embedding_params_;                                                 \
  }
}  // namespace HugeCTR
