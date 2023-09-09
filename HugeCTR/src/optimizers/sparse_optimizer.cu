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
#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/utils.cuh"
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_run_length_encode.cuh"
#include "cub/device/device_scan.cuh"

namespace HugeCTR {

template <typename TypeHashKey, typename TypeEmbeddingComp>
EmbeddingOptimizer<TypeHashKey, TypeEmbeddingComp>::EmbeddingOptimizer(
    size_t max_vocabulary_size_per_gpu_, SparseEmbeddingHashParams &param,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>> &buf)
    : param(param) {
  // new optimizer params used by update_params
  switch (param.opt_params.optimizer) {
    case Optimizer_t::Adam:  // adam
    {
      {
        buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                     &opt_tensors_.opt_m_tensors_);
        buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                     &opt_tensors_.opt_v_tensors_);
      }
      if (param.opt_params.update_type == Update_t::LazyGlobal) {
        buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                     &opt_tensors_.opt_prev_time_tensors_);
      }
      break;
    }
    case Optimizer_t::AdaGrad:  // nesterov
    {
      buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                   &opt_tensors_.opt_accm_tensors_);
      break;
    }
    case Optimizer_t::MomentumSGD:  // momentum_sgd
    {
      buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                   &opt_tensors_.opt_momentum_tensors_);
      break;
    }

    case Optimizer_t::Nesterov:  // nesterov
    {
      buf->reserve({max_vocabulary_size_per_gpu_, param.embedding_vec_size},
                   &opt_tensors_.opt_accm_tensors_);
      break;
    }

    case Optimizer_t::SGD:
      break;

    default:
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
  }

  { buf->reserve({1, param.get_batch_size(true) * param.max_feature_num}, &sample_id_tensors_); }
  {
    buf->reserve({1, param.get_batch_size(true) * param.max_feature_num}, &sample_id_sort_tensors_);
  }
  {
    buf->reserve({1, param.get_batch_size(true) * param.max_feature_num},
                 &hash_value_index_sort_tensors_);
  }
  {
    buf->reserve({1, param.get_batch_size(true) * param.max_feature_num + 1},
                 &hash_value_index_count_offset_tensors_);
  }
  {
    buf->reserve({1, param.get_batch_size(true) * param.max_feature_num},
                 &new_hash_value_flag_tensors_);
  }
  {
    buf->reserve({1, param.get_batch_size(true) * param.max_feature_num},
                 &hash_value_flag_sumed_tensors_);
  }
  { buf->reserve({1, 1}, &hash_value_index_count_counter_tensors_); }
  {
    // cal the temp storage bytes for CUB radix sort
    size_t size = 0;
    cub::DeviceRadixSort::SortPairs((void *)nullptr, size, (size_t *)nullptr, (size_t *)nullptr,
                                    (TypeHashKey *)nullptr, (TypeHashKey *)nullptr,
                                    param.get_batch_size(true) * param.max_feature_num);

    // new temp storage tensors for CUB radix sort
    buf->reserve({size}, &temp_storage_sort_tensors_);
  }

  {
    size_t size = 0;
    cub::DeviceScan::InclusiveSum((void *)nullptr, size, (uint32_t *)nullptr, (uint32_t *)nullptr,
                                  param.get_batch_size(true) * param.max_feature_num);

    buf->reserve({size}, &temp_storage_scan_tensors_);
  }
}

template <typename TypeHashKey, typename TypeEmbeddingComp>
void EmbeddingOptimizer<TypeHashKey, TypeEmbeddingComp>::initialize(const GPUResource &local_gpu) {
  switch (param.opt_params.optimizer) {
    case Optimizer_t::Adam:  // adam
      CK_CUDA_THROW_(cudaMemsetAsync(opt_tensors_.opt_m_tensors_.get_ptr(), 0,
                                     opt_tensors_.opt_m_tensors_.get_size_in_bytes(),
                                     local_gpu.get_stream()));
      CK_CUDA_THROW_(cudaMemsetAsync(opt_tensors_.opt_v_tensors_.get_ptr(), 0,
                                     opt_tensors_.opt_v_tensors_.get_size_in_bytes(),
                                     local_gpu.get_stream()));
      param.opt_params.hyperparams.adam.times = 0;
      if (param.opt_params.update_type == Update_t::LazyGlobal) {
        dim3 grid(local_gpu.get_sm_count() * 4, 1, 1);
        dim3 block(512, 1, 1);
        initialize_array<<<grid, block, 0, local_gpu.get_stream()>>>(
            opt_tensors_.opt_prev_time_tensors_.get_ptr(),
            opt_tensors_.opt_prev_time_tensors_.get_num_elements(), uint64_t(1));
      }
      break;
    case Optimizer_t::AdaGrad:
      CK_CUDA_THROW_(cudaMemsetAsync(opt_tensors_.opt_accm_tensors_.get_ptr(),
                                     param.opt_params.hyperparams.adagrad.initial_accu_value,
                                     opt_tensors_.opt_accm_tensors_.get_size_in_bytes(),
                                     local_gpu.get_stream()));
      break;
    case Optimizer_t::MomentumSGD:  // momentum_sgd
      CK_CUDA_THROW_(cudaMemsetAsync(opt_tensors_.opt_momentum_tensors_.get_ptr(), 0,
                                     opt_tensors_.opt_momentum_tensors_.get_size_in_bytes(),
                                     local_gpu.get_stream()));
      break;

    case Optimizer_t::Nesterov:  // nesterov
      CK_CUDA_THROW_(cudaMemsetAsync(opt_tensors_.opt_accm_tensors_.get_ptr(), 0,
                                     opt_tensors_.opt_accm_tensors_.get_size_in_bytes(),
                                     local_gpu.get_stream()));
      break;

    case Optimizer_t::SGD:
      break;

    default:
      throw std::runtime_error(
          std::string("[HCDEBUG][ERROR] Runtime error: Invalid optimizer type\n"));
  }
}

namespace {

__global__ void value_count_kernel_2(int nnz, const uint32_t *new_hash_value_flag,
                                     const uint32_t *hash_value_flag_sumed,
                                     uint32_t *hash_value_index_index, uint32_t *counter)

{
  // 遍历grid，但是需要小于该batch的非零key数目，其实就是 hash_table_value 的行数
  for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x * gridDim.x) {
    uint32_t flag = new_hash_value_flag[gid];
    if (flag == 1) {
      // 设定
      hash_value_index_index[hash_value_flag_sumed[gid] - 1] = gid;
    }
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
    *counter = hash_value_flag_sumed[nnz - 1];
    hash_value_index_index[*counter] = nnz;
  }
}

/*
6.3 拓展sample id
这里对应了第一步，在后续代码之中，每个key对应了一个sample ID。
总体思路就是找到每个 key（sample ID） 和梯度矩阵，或者说和embedding_feature之中哪一行相对应，
我们后续就直接以 embedding_feature来看，暂时不考虑梯度矩阵 。
可以大致理解为把样本id扩展为key id的列表。
        step1: expand sample IDs, calling sample_id_expand_kernel()
就是调用 sample_id_expand_kernel 来拓展sample id。这里 sample_id 是成员变量 sample_id_tensors_的引用，这样就可以直接修改成员变量。
        Tensor2<TypeHashKey> sample_id_tensors_; //< The temp memory to store the sample ids of hash table value in update_params().
具体代码如下：
        Tensor2<TypeHashKey> &sample_id = sample_id_tensors_;

        // step1: expand sample IDs
        block_size = 64;
        grid_size = (batch_size * slot_num - 1) / block_size + 1;
        sample_id_expand_kernel<<<grid_size, block_size, 0, stream>>>(
            batch_size, slot_num, row_offset.get_ptr(), sample_id.get_ptr());
通过前面分析我们知道，embedding vector个数是：batch_size x slot_num，也就是说，CSR 有几行，这里就有几个向量。
 所以这里就直接读取CSR行信息即可。
 即， sample_id_expand_kernel 会把 sample_id_tensors_ 设置为 CSR row offset（expand sample id by row_offset），就是找到 CSR row offset 之中的index。

CSR row_offset = [0,4,7,9,10]，样本之中key的数值是40,50,10,20,30,50,10,30,20,10，
那么 40,50,10,20对应了 0，30,50,10对应了1，30,20对应了 2，10对应了3。
因此，sample_id 数值是 [0,0,0,0,1,1,1,2,2,3]，就是记录了该 batch 在 embedding_feature_tensors_ 之中的 row index。

sample_id_expand_kernel 代码如下，这里几个重点：
      gid 是grid ID，表示本线程对应了embedding_feature_tensors_ 哪个元素。

      blockIdx 表示一个样本。

      (batch_size * slot_num) 代表 本batch在 嵌入层输出 train_output_tensors_ 之中对应了多少行，
      或者说是在 embedding_feature_tensors_ 之中占据了多少行，其实 embedding_feature_tensors_ 也就这么大。

      sample_id[offset + i] = gid; 目的就是记录该样本某key在 embedding_feature_tensors_ 之中的 row index（对应哪一行）。
      embedding_feature_tensors_ 这个稠密向量是由 hash_table_value 之中"CSR 本行的元素数目"个稠密向量做pooling得到的结果。

我们把目前涉及的变量整理如下，这里假定从CSR数值到hash_value_index_tensors_ 行的映射是取十位数，比如50就映射到第5行。

名称	数值	意义
CSR row offset	0,4,7,9,10	两个样本，两个slot，所以分成四行
CSR value	40,50,10,20,30,50,10,30,20,10	样本内容
hash_value_index_tensors_	4,5,1,2,3,5,1,3,2,1	低维嵌入表的index，样本每个key对应一个，比如50对应了 hash_table_value 第5行
hash_table_value	5 x 8 的矩阵	低维嵌入表，假定稠密向量长度是8，因为一共只有5个不同数字，所以只有5行
embedding_feature_tensors_	4 x 8 的矩阵	嵌入层输出的稠密向量。形状是(batch_size * slot_num) * embedding_vec_len
sample_id	0,0,0,0,1,1,1,2,2,3	每个样本的每个key 对应了embedding_feature_tensors_ 中的 row index。比如CSR第一行是40,50,10,20，它们都为 embedding_feature_tensors_ 的第一行做出了贡献。
 * */
// expand sample id by row_offset
template <typename TypeKey>
__global__ void sample_id_expand_kernel(int batch_size, int slot_num, const TypeKey *row_offset,
                                        TypeKey *sample_id) {
  //// 本线程对应的grid id，其实对应的就是global thread id
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < (batch_size * slot_num)) {  // 假如batch_size=2，slot_num=2，取值为 gid < 4
     // 并不是每个GPU线程都会走到这里，对应我们的假设，则只会取出gid = 0~3 这样的线程才会进行下面配置操作
    // 比如，假定gid取值范围8，那么只有gid=0,gid=1,gid=2,gid=3 这几个线程会进入if，执行操作，其余线程不会进入，比如grid=4就不会进入
    TypeKey offset = row_offset[gid];  // 拿到对应的个数，比如 row_offset[0]，row_offset[1]，row_offset[2]的数值
    int value_num = row_offset[gid + 1] - offset;  // 拿到CSR 本行的元素数目
    for (int i = 0; i < value_num; i++) {
      sample_id[offset + i] = gid;  // 记录该样本某key在 embedding_feature_tensors_ 之中的 row index
    }
  }
}

__global__ void value_count_kernel_1(int nnz, const size_t *hash_value_index_sort,
                                     uint32_t *new_hash_value_flag) {
  for (int gid = blockIdx.x * blockDim.x + threadIdx.x; gid < nnz; gid += blockDim.x * gridDim.x) {
    size_t cur_value = hash_value_index_sort[gid];
    if (gid > 0) {
      size_t former_value = hash_value_index_sort[gid - 1];
      // decide if this is the start of a group(the elements in this group have the same
      // hash_value_index_sort)
      if (cur_value != former_value) {
        new_hash_value_flag[gid] = 1;
      } else {
        new_hash_value_flag[gid] = 0;
      }
    } else {  // gid == 0
      new_hash_value_flag[gid] = 1;
    }
  }
}

// Helper function to accumulate the weight gradients for a thread's embedding vector
template <typename TypeKey, typename TypeEmbeddingComp>
__device__ __forceinline__ float accumulate_gradients(int embedding_vec_size,
                                                      const TypeKey *sample_id,
                                                      const uint32_t *hash_value_index_count_offset,
                                                      const TypeEmbeddingComp *wgrad, float scaler,
                                                      uint32_t offset, int bid, int tid) {
  // 哪一行更新几次
  // 如果bid=0,则sum_num = hash_value_index_count_offset[1] - hash_value_index_count_offset[0] = 3 - 0 = 3个。
  // bid对应了key，比如 40,50,10,20,30,50,10,30,20,10 这些key，其key就是10～50这个5个。
  // 所以 bid = 0 就是要更新10对应的低维矩阵稠密向量，就是hash_table_value[0]这一行，有三个1，应该更新3次
  uint32_t sample_num = hash_value_index_count_offset[bid + 1] - hash_value_index_count_offset[bid];

  // 计算梯度
  float gi = 0.0f;

  // sample_id_sort [0,1,3,0,2,1,2,0,0,1] ---- 第几行，恰恰和 wgrad 对上了
  for (int i = 0; i < sample_num; i++) {   // offset 就是0, 3, 5, 7, 8，比如对于第1行，需要更新3次
    // sample_id 是[0,1,3,0,2,1,2,0,0,1]，对应了低维矩阵第1,2,4,...,行，就是3个10分别在输出稠密向量的哪一行
    // 更新这几次，就是一个累积，这个累积用哪些梯度来累积。
    int sample_index = sample_id[offset + i];   // 找到本样本梯度
    gi += TypeConvertFunc<float, TypeEmbeddingComp>::convert(
        wgrad[sample_index * embedding_vec_size + tid]);  // 本线程梯度，并且累积
  }
  return gi / scaler;
}

// First step of the global update with the Adam optimizer: compute gradient and add the
// corresponding terms to the moving-average accumulators
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adam_kernel_global(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                       const AdamOptHyperParams adam, TypeEmbeddingComp *m_ptr,
                                       TypeEmbeddingComp *v_ptr, const TypeKey *sample_id,
                                       const size_t *hash_value_index_sort,
                                       const uint32_t *hash_value_index_count_offset,
                                       const TypeEmbeddingComp *wgrad, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(m_ptr[feature_index]) +
               (1.0f - adam.beta1) * gi / adam.beta1;
    float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(v_ptr[feature_index]) +
               (1.0f - adam.beta2) * gi * gi / adam.beta2;

    m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
  }
}

// Second step of the global update with the Adam optimizer: update the moving-average accumulators
// and the weights for all the features
template <typename TypeEmbeddingComp>
__global__ void adam_update_kernel_global(int embedding_vec_size,
                                          size_t table_size,  // vocabulary size / factor
                                          const AdamOptHyperParams adam, TypeEmbeddingComp *m_ptr,
                                          TypeEmbeddingComp *v_ptr, float alpha_t,
                                          float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float mi =
        adam.beta1 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(m_ptr[feature_index]);
    float vi =
        adam.beta2 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(v_ptr[feature_index]);

    m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);

    float weight_diff = -alpha_t * mi / (sqrtf(vi) + adam.epsilon);
    hash_table_value[feature_index] += weight_diff;
  }
}

// First step of the global update with Momentum SGD: compute gradient and add the corresponding
// term to the momentum
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_momentum_sgd_kernel_global(
    uint32_t hash_value_index_count_num, int embedding_vec_size, float lr,
    const MomentumSGDOptHyperParams momentum, TypeEmbeddingComp *momentum_ptr,
    const TypeKey *sample_id, const size_t *hash_value_index_sort,
    const uint32_t *hash_value_index_count_offset, const TypeEmbeddingComp *wgrad, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mo = TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum_ptr[feature_index]) -
               lr * gi / momentum.factor;
    momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
  }
}

// Second step of the global update with Momentum SGD: update the momentum and the weights for all
// the features
template <typename TypeEmbeddingComp>
__global__ void momentum_sgd_update_kernel_global(int embedding_vec_size,
                                                  size_t table_size,  // vocabulary size / factor
                                                  const MomentumSGDOptHyperParams momentum,
                                                  TypeEmbeddingComp *momentum_ptr,
                                                  float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float mo = TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum_ptr[feature_index]);
    mo *= momentum.factor;
    hash_table_value[feature_index] += mo;
    momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);
  }
}

// First step of the global update with Nesterov: update momentum and weights for all the features
template <typename TypeEmbeddingComp>
__global__ void nesterov_global_update_kernel_global(int embedding_vec_size,
                                                     size_t table_size,  // vocabulary size / factor
                                                     const NesterovOptHyperParams nesterov,
                                                     TypeEmbeddingComp *accm_ptr,
                                                     float *hash_table_value) {
  const int TILE_SIZE = blockDim.x * gridDim.x;
  for (size_t feature_index = blockIdx.x * blockDim.x + threadIdx.x;
       feature_index < table_size * embedding_vec_size; feature_index += TILE_SIZE) {
    float accm = TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]);
    accm *= nesterov.mu;
    accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);
    hash_table_value[feature_index] += accm * nesterov.mu;
  }
}

// Second step of the global update with Nesterov: compute gradient, add the corresponding term
// to the momentum and update the weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void nesterov_local_update_kernel_global(
    uint32_t hash_value_index_count_num, int embedding_vec_size, float lr,
    const NesterovOptHyperParams nesterov, TypeEmbeddingComp *accm_ptr, const TypeKey *sample_id,
    const size_t *hash_value_index_sort, const uint32_t *hash_value_index_count_offset,
    const TypeEmbeddingComp *wgrad, float *hash_table_value, float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float accm = TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]);
    accm -= lr * gi;
    accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm);
    hash_table_value[feature_index] -= (1 + nesterov.mu) * (lr * gi);
  }
}

// Local update for the Adam optimizer: compute the gradients and update the accumulators and the
// weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adam_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                const AdamOptHyperParams adam, TypeEmbeddingComp *m_ptr,
                                TypeEmbeddingComp *v_ptr, float alpha_t, const TypeKey *sample_id,
                                const size_t *hash_value_index_sort,
                                const uint32_t *hash_value_index_count_offset,
                                const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mi =
        adam.beta1 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(m_ptr[feature_index]) +
        (1.0f - adam.beta1) * gi;
    float vi =
        adam.beta2 * TypeConvertFunc<float, TypeEmbeddingComp>::convert(v_ptr[feature_index]) +
        (1.0f - adam.beta2) * gi * gi;
    m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
    float weight_diff = -alpha_t * mi / (sqrtf(vi) + adam.epsilon);

    hash_table_value[feature_index] += weight_diff;
  }
}

//其本质就是更新 hash_table_value，也就是嵌入层的权重。具体我们后文会结合反向传播进行分析。
// Local update for the Adagrad optimizer: compute the gradients and update the accumulators and the
// weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adagrad_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                   float lr, const AdaGradParams adagrad,
                                   TypeEmbeddingComp *accum_ptr, const TypeKey *sample_id,
                                   const size_t *hash_value_index_sort,
                                   const uint32_t *hash_value_index_count_offset,
                                   const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                   float scaler) {
  int bid = blockIdx.x;  // 一个block对应一个样本之中的一个key，比如例子之中的30
  int tid = threadIdx.x; // 本线程

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    // 找到本线程样本在 hash_value_index_sort 的偏移
    uint32_t offset = hash_value_index_count_offset[bid];  // [0, 3, 5, 7, 8, 0, 0, 0, 0, 0]

    // 累积得出梯度
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    // 找到本样本在低维矩阵之中的row index
    size_t row_index = hash_value_index_sort[offset];

    // 注意，hash_table_value 是元素级别，比如稠密向量长度是8，那么在 hash_table_value 里面就有8个元素
    // feature_index 就是得到本线程对应的 embedding vector 之中的哪个元素
    size_t feature_index = row_index * embedding_vec_size + tid;

    //accum_ptr 来自优化器
    float accum =
        TypeConvertFunc<float, TypeEmbeddingComp>::convert(accum_ptr[feature_index]) + gi * gi;

    accum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accum);
    float weight_diff = -lr * gi / (sqrtf(accum) + adagrad.epsilon);

    hash_table_value[feature_index] += weight_diff;  // 更新权重 // 更新梯度
  }
}

// Local update for Momentum SGD: compute the gradients and update the momentum and the weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_momentum_sgd_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                        float lr, const MomentumSGDOptHyperParams momentum,
                                        TypeEmbeddingComp *momentum_ptr, const TypeKey *sample_id,
                                        const size_t *hash_value_index_sort,
                                        const uint32_t *hash_value_index_count_offset,
                                        const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                        float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float mo = momentum.factor *
                   TypeConvertFunc<float, TypeEmbeddingComp>::convert(momentum_ptr[feature_index]) -
               lr * gi;
    momentum_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mo);

    hash_table_value[feature_index] += mo;
  }
}

// Local update for Nesterov: compute the gradients and update the accumulators and the weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_nesterov_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                    float lr, const NesterovOptHyperParams nesterov,
                                    TypeEmbeddingComp *accm_ptr, const TypeKey *sample_id,
                                    const size_t *hash_value_index_sort,
                                    const uint32_t *hash_value_index_count_offset,
                                    const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                    float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;
    float accm_old = TypeConvertFunc<float, TypeEmbeddingComp>::convert(accm_ptr[feature_index]);
    float accm_new = nesterov.mu * accm_old - lr * gi;
    accm_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(accm_new);
    float weight_diff = -nesterov.mu * accm_old + (1.0f + nesterov.mu) * accm_new;

    hash_table_value[feature_index] += weight_diff;
  }
}

// Local update for SGD: compute the gradients and update the weights
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_sgd_kernel(uint32_t hash_value_index_count_num, int embedding_vec_size,
                               float lr, const TypeKey *sample_id,
                               const size_t *hash_value_index_sort,
                               const uint32_t *hash_value_index_count_offset,
                               const TypeEmbeddingComp *wgrad, float *hash_table_value,
                               float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    float weight_diff = -lr * gi;

    size_t feature_index = row_index * embedding_vec_size + tid;
    hash_table_value[feature_index] += weight_diff;
  }
}

// Lazy global update for the Adam optimizer: compute the gradients and update the weights and the
// accumulators (local approximation of the global update)
template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_adam_kernel_lazy(uint32_t hash_value_index_count_num, int embedding_vec_size,
                                     const AdamOptHyperParams adam, uint64_t *prev_time_ptr,
                                     TypeEmbeddingComp *m_ptr, TypeEmbeddingComp *v_ptr,
                                     float alpha_t_common, uint64_t times, const TypeKey *sample_id,
                                     const size_t *hash_value_index_sort,
                                     const uint32_t *hash_value_index_count_offset,
                                     const TypeEmbeddingComp *wgrad, float *hash_table_value,
                                     float scaler) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < hash_value_index_count_num) {
    uint32_t offset = hash_value_index_count_offset[bid];
    float gi = accumulate_gradients(embedding_vec_size, sample_id, hash_value_index_count_offset,
                                    wgrad, scaler, offset, bid, tid);

    size_t row_index = hash_value_index_sort[offset];
    size_t feature_index = row_index * embedding_vec_size + tid;

    // First update the weights
    uint64_t prev_time = prev_time_ptr[feature_index];
    prev_time_ptr[feature_index] = times;
    uint64_t skipped = times - prev_time;
    float beta1_pow_skipped = powf(adam.beta1, skipped);
    float alpha_t = alpha_t_common * sqrtf(1.0f - powf(adam.beta2, prev_time)) /
                    (1.0f - powf(adam.beta1, prev_time)) * (1.0f - beta1_pow_skipped);
    float mi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(m_ptr[feature_index]);
    float vi = TypeConvertFunc<float, TypeEmbeddingComp>::convert(v_ptr[feature_index]);
    float weight_diff = -alpha_t * mi / (sqrtf(vi) + adam.epsilon);
    hash_table_value[feature_index] += weight_diff;

    // Then update the moving-average accumulators
    mi = beta1_pow_skipped * mi + (1.0f - adam.beta1) * gi;
    vi = powf(adam.beta2, skipped) * vi + (1.0f - adam.beta2) * gi * gi;
    m_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(mi);
    v_ptr[feature_index] = TypeConvertFunc<TypeEmbeddingComp, float>::convert(vi);
  }
}

template <typename TypeKey, typename TypeEmbeddingComp>
__global__ void opt_sgd_atomic_kernel(int nnz, int embedding_vec_size, float lr_scale,
                                      const size_t *hash_value_index, const TypeKey *sample_ids,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < nnz) {
    for (int key_id = bid; key_id < nnz; key_id += gridDim.x) {
      int sample_id = sample_ids[key_id];
      float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                     wgrad[sample_id * embedding_vec_size + tid]);

      // atomic update
      size_t value_index = hash_value_index[key_id];
      size_t feature_index = value_index * embedding_vec_size + tid;
      atomicAdd(&hash_table_value[feature_index], deltaw);
    }
  }
}

// only support LocalizedSlotSparseEmbeddingOneHot
template <typename TypeEmbeddingComp>
__global__ void opt_sgd_atomic_kernel(int nnz, int embedding_vec_size, float lr_scale,
                                      const size_t *hash_value_index,
                                      const TypeEmbeddingComp *wgrad, float *hash_table_value) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid < embedding_vec_size && bid < nnz) {
    for (int key_id = bid; key_id < nnz; key_id += gridDim.x) {
      // for one-hot, the max_feature_per_slot is 1, so sample_id is equal to key_id
      float deltaw = -lr_scale * TypeConvertFunc<float, TypeEmbeddingComp>::convert(
                                     wgrad[key_id * embedding_vec_size + tid]);

      // atomic update
      size_t value_index = hash_value_index[key_id];
      size_t feature_index = value_index * embedding_vec_size + tid;
      atomicAdd(&hash_table_value[feature_index], deltaw);
    }
  }
}

}  // namespace
//6.2 更新
//    其内部主要是通过 opt_adagrad_kernel 进行更新。
//6.2.2 update代码
//    我们摘录 EmbeddingOptimizer::update 的代码如下，
//        这里只是选择了Optimizer_t::AdaGrad相关部分，其通过 opt_adagrad_kernel 进行更新。
//        这里可以清楚看到注释中的各个步骤，我们接下来就会逐一分析。
template <typename TypeHashKey, typename TypeEmbeddingComp>
void EmbeddingOptimizer<TypeHashKey, TypeEmbeddingComp>::update(
    size_t batch_size, size_t slot_num, size_t embedding_vec_size,
    size_t max_vocabulary_size_per_gpu, size_t nnz, const Tensor2<TypeHashKey> &row_offset,
    Tensor2<size_t> &hash_value_index, const Tensor2<TypeEmbeddingComp> &wgrad,
    Tensor2<float> &hash_table_value, size_t sm_count, cudaStream_t stream) {
  OptimizerTensor<TypeEmbeddingComp> &opt_tensor = opt_tensors_;
  OptParams &opt_params = param.opt_params;
  Tensor2<TypeHashKey> &sample_id = sample_id_tensors_;
  Tensor2<TypeHashKey> &sample_id_sort = sample_id_sort_tensors_;
  Tensor2<size_t> &hash_value_index_sort = hash_value_index_sort_tensors_;
  Tensor2<uint32_t> &hash_value_index_count_offset = hash_value_index_count_offset_tensors_;
  Tensor2<uint32_t> &new_hash_value_flag = new_hash_value_flag_tensors_;
  Tensor2<uint32_t> &hash_value_flag_sumed = hash_value_flag_sumed_tensors_;
  Tensor2<uint32_t> &hash_value_index_count_counter = hash_value_index_count_counter_tensors_;
  Tensor2<void> &temp_storage_sort = temp_storage_sort_tensors_;
  Tensor2<void> &temp_storage_scan = temp_storage_scan_tensors_;

  if (slot_num == 0) {
    return;
  }

  size_t block_size, grid_size;

  try {
    // step1: expand sample IDs
    block_size = 64;
    grid_size = (batch_size * slot_num - 1) / block_size + 1;
    sample_id_expand_kernel<<<grid_size, block_size, 0, stream>>>(
        batch_size, slot_num, row_offset.get_ptr(), sample_id.get_ptr());

    if (opt_params.optimizer == Optimizer_t::SGD &&
        opt_params.hyperparams.sgd.atomic_update) {  // for SGD, do atomic update
      const size_t block_size = embedding_vec_size;
      const size_t grid_size = min(max(1ul, nnz), sm_count * 32);

      float lr_scale = opt_params.lr / opt_params.scaler;
      opt_sgd_atomic_kernel<<<grid_size, block_size, 0, stream>>>(
          nnz, embedding_vec_size, lr_scale, hash_value_index.get_ptr(), sample_id.get_ptr(),
          wgrad.get_ptr(), hash_table_value.get_ptr());
    } else {
/*
    6.5 排序
      这部分对应第三步：
      step3: sort by value_index (will call cub::DeviceRadixSort::SortPairs in cub lib)
      现在得到了：sample_id 数值是 [0,0,0,0,1,1,1,2,2,3]，就是记录了该 batch 在 embedding_feature_tensors_ 之中的 row index。
      就是把 sample_id 按照 hash_value_index 来排序，最后排序结果放入 hash_value_index_sort 和 sample_id_sort。在我们例子之中，得到结果如下：hash_value_index_sort 是 [1,1,1,2,2,3,3,4,5,5]。sample_id_sort 是 [0,1,3,0,2,1,2,0,0,1 ]。
      我们还是用表格记录：
名称	数值	意义
CSR row offset	0,4,7,9,10	两个样本，两个slot，所以分成四行
CSR value	40,50,10,20,30,50,10,30,20,10	样本内容
hash_value_index_tensors_	4,5,1,2,3,5,1,3,2,1	低维嵌入表的index，样本每个key对应一个，比如50对应了 hash_table_value 第5行
hash_table_value	5 x 8 的矩阵	低维嵌入表，假定稠密向量长度是8，因为一共只有5个不同数字，所以只有5行
embedding_feature_tensors_	4 x 8 的矩阵	嵌入层输出的稠密向量。形状是(batch_size * slot_num) * embedding_vec_len
sample_id	0,0,0,0,1,1,1,2,2,3	每个样本的每个key 对应了embedding_feature_tensors_ 中的 row index。比如CSR第一行是40,50,10,20，它们都为 embedding_feature_tensors_ 的第一行做出了贡献。
sample_id_sort	[0,1,3,0,2,1,2,0,0,1 ]	和 hash_value_index_sort 对应，就是 hash_value_index_sort 前三个 1 分别对应了embedding_feature 的第1行，第2行，第4行（从0开始的序列）
hash_value_index_sort	[1,1,1,2,2,3,3,4,5,5]	排序之后的结果，举例来说，111 意思是本batch之中，一共有3个key对最终embedding_feature第一行做出了贡献
具体代码如下：
 * */
      // step3: sort by hash_value_index   具体使用方法如下：007-006.png
      int end_bit = static_cast<int>(log2(static_cast<float>(max_vocabulary_size_per_gpu))) + 1;
      size_t temp_storage_sort_size = temp_storage_sort.get_size_in_bytes();
      CK_CUDA_THROW_(cub::DeviceRadixSort::SortPairs(
          temp_storage_sort.get_ptr(), temp_storage_sort_size, hash_value_index.get_ptr(),
          hash_value_index_sort.get_ptr(), sample_id.get_ptr(), sample_id_sort.get_ptr(), nnz, 0,
          end_bit, stream, false));

      /*
6.6 计算value_index对应的数目
现在知道了 hash_value_index_sort 是 [1,1,1,2,2,3,3,4,5,5]，sample_id_sort 是 [0,1,3,0,2,1,2,0,0,1 ]。
      hash_value_index_sort 是hash_value_index排序之后的结果，举例来说，111 意思是本batch之中，一共有3个key对最终embedding_feature第一行做出了贡献
      sample_id_sort 和 hash_value_index_sort 对应，就是 hash_value_index_sort 前三个 1 分别对应了embedding_feature 的第1行，第2行，第4行（从0开始的序列）
接下来需要知道 embedding_feature_tensors_ 每行的来源是多少个 hash_table_value 行，
比如第0行有4个，第1行有3个......。embedding_feature_tensors_ 之中的一个行 是被同一个slot的多个 hash_table_value 行的稠密向量做pooling完成的。

就是对 hash_value_index_sort 进行处理，这里是 embedding 表 hash_table_value 的 row index。

我们接下来一点点分析。
6.6.1 value_count_kernel_1
value_count_kernel_1目的是找到新的group，就是新的 row index。
目的是为了计算每个row index对应的sample id 个数。就是找到哪些点是新行起始点。我们拓展表格如下。

名称	数值	意义
CSR row offset	0,4,7,9,10	两个样本，两个slot，所以分成四行
CSR value	40,50,10,20,30,50,10,30,20,10	样本内容
hash_value_index_tensors_	4,5,1,2,3,5,1,3,2,1	低维嵌入表的index，样本每个key对应一个，比如50对应了 hash_table_value 第5行
sample_id	0,0,0,0,1,1,1,2,2,3	每个样本的每个key 对应了embedding_feature_tensors_ 中的 row index。比如CSR第一行是40,50,10,20，它们都为 embedding_feature_tensors_ 的第一行做出了贡献。
sample_id_sort	[0,1,3,0,2,1,2,0,0,1 ]	和 hash_value_index_sort 对应，就是 hash_value_index_sort 前三个 1 分别对应了 embedding_feature 的第1行，第2行，第4行（从0开始的序列）
hash_value_index_sort	[1,1,1,2,2,3,3,4,5,5]	排序之后的结果，举例来说，1,1,1 意思是本batch之中，一共有3个key对最终embedding_feature第一行做出了贡献
new_hash_value_flag	[1,0,0,1,0,1,0,1,1,0]	为了计算每个row index对应的sample id 个数。就是找到哪些点是新行起始点
       * */
      // step4: count the number for each unduplicated hash_value_index
      CK_CUDA_THROW_(
          cudaMemsetAsync(hash_value_index_count_counter.get_ptr(), 0, sizeof(uint32_t), stream));

      constexpr size_t max_grid_size = 384;
      block_size = 256;
      grid_size = min(max_grid_size, (nnz - 1) / block_size + 1);

      //// 目的是找到新的group，就是新的 row index。目的是为了计算每个row index对应的sample id个数
      value_count_kernel_1<<<grid_size, block_size, 0, stream>>>(
          nnz, hash_value_index_sort.get_ptr(), new_hash_value_flag.get_ptr());

      /*
6.6.2 prefix_sum
对 new_hash_value_flag 排序，目的是得到每个group（row index）内部包含多少元素，放入 hash_value_flag_sumed 之中。
这里使用了 cub::DeviceScan::InclusiveSum，如果想深入研究，可以参见 https://nvlabs.github.io/cub/structcub_1_1_device_scan.html 。

以下是函数说明  007-007  以下是使用方法 007-008
我们拓展表格如下。
名称	数值	意义
CSR row offset	0,4,7,9,10	两个样本，两个slot，所以分成四行
CSR value	40,50,10,20,30,50,10,30,20,10	样本内容
hash_value_index_tensors_	[4,5,1,2,3,5,1,3,2,1]	低维嵌入表的index，样本每个key对应一个，比如50对应了 hash_table_value 第5行
sample_id	[0,0,0,0,1,1,1,2,2,3]	每个样本的每个key 对应了embedding_feature_tensors_ 中的 row index。比如CSR第一行是40,50,10,20，它们都为 embedding_feature_tensors_ 的第一行做出了贡献。
sample_id_sort	[0,1,3,0,2,1,2,0,0,1]	和 hash_value_index_sort 对应，就是 hash_value_index_sort 前三个 1 分别对应了 embedding_feature 的第1行，第2行，第4行（从0开始的序列）
hash_value_index_sort	[1,1,1,2,2,3,3,4,5,5]	排序之后的结果，举例来说，1,1,1 意思是本batch之中，一共有3个key对最终embedding_feature第一行做出了贡献
new_hash_value_flag	[1,0,0,1,0,1,0,1,1,0]	为了计算每个row index对应的sample id 个数。就是找到哪些点是新行起始点
hash_value_flag_sumed	[1,1,1,2,2,3,3,4,5,5]	对 new_hash_value_flag 合并，目的是得到每个group（row index）内部包含多少元素。
hash_table_value	5 x 8 的矩阵	低维嵌入表，假定稠密向量长度是8，因为一共只有5个不同数字，所以只有5行
       */
      // prefix_sum
      size_t temp_storage_scan_size = temp_storage_scan.get_size_in_bytes();
      CK_CUDA_THROW_(cub::DeviceScan::InclusiveSum(
          temp_storage_scan.get_ptr(), temp_storage_scan_size, new_hash_value_flag.get_ptr(),
          hash_value_flag_sumed.get_ptr(), nnz, stream));

      /*
6.6.3 value_count_kernel_2
这个代码作用就是得到最终每行元素个数。

hash_hash_value_index_count_num 是index总数，就是一共真实有几行，其对应了nnz。
      * @param nnz non-zero feature number per batch

现在知道了 hash_value_index_sort 是 [1,1,1,2,2,3,3,4,5,5]，sample_id_sort 是 [0,1,3,0,2,1,2,0,0,1 ]，
new_hash_value_flag 是 [1,0,0,1,0,1,0,1,1,0]，里面放置了本行是不是新行。
hash_value_flag_sumed 是[ 1,1,1,2,2,3,3,4,5,5 ]。

我们分析一下代码。总体思想是：在 hash_value_index_index（对应传进来的参数是 hash_value_index_count_offset）设定 "按照数目计算的，
对应的 embedding 表 index（就是对应的 embedding 表行号）"。
因为embedding_feature 最多只有5行（nnz个数），所以这里取前五个即可。

比如，每个block要处理低维稠密矩阵一行。如 bid = 1，它希望更新低维稠密矩阵第2行，但是想知道更新几次。
所以先从 hash_value_index_count_offset[1] 得到了数值 3，然后找到 hash_value_index_sort[3] 来进行处理。

具体是：遍历grid，但是需要小于nnz（该batch的非零key数目），其实就是 hash_table_value 的行数。
比如说nnz这里等于10，gid 取值就是0～9。grid为0，3，5，7，8 时候new_hash_value_flag[gid] 为 1。
hash_value_flag_sumed[gid]分别为：1,2,3,4,5。
所以 hash_value_index_count_offset 是 [0, 3, 5, 7, 8, 0, 0, 0, 0, 0]，
这些是 hash_value_index_sort 之中的offset。

到目前为止，所有变量如下：
名称	数值	意义
CSR row offset	0,4,7,9,10	两个样本，两个slot，所以分成四行
CSR value	40,50,10,20,30,50,10,30,20,10	样本内容
hash_table_value	5 x 8 的矩阵	低维嵌入表，假定稠密向量长度是8，因为一共只有5个不同数字（nnz），所以只有5行
embedding_feature_tensors_	4 x 8 的矩阵	嵌入层输出的稠密向量。形状是(batch_size * slot_num) * embedding_vec_len
hash_value_index_tensors_	[4,5,1,2,3,5,1,3,2,1]	低维嵌入表的index，样本每个key对应一个，比如50对应了 hash_table_value 第5行
sample_id	[0,0,0,0,1,1,1,2,2,3]	每个样本的每个key 对应了embedding_feature_tensors_ 中的 row index。比如CSR第一行是40,50,10,20，它们都为 embedding_feature_tensors_ 的第一行做出了贡献。
sample_id_sort	[0,1,3,0,2,1,2,0,0,1]	和 hash_value_index_sort 对应，就是 hash_value_index_sort 前三个 1 分别对应了 embedding_feature 的第1行，第2行，第4行（从0开始的序列）
hash_value_index_sort	[1,1,1,2,2,3,3,4,5,5]	排序之后的结果，举例来说，1,1,1 意思是本batch之中，一共有3个key对最终embedding_feature第一行做出了贡献
new_hash_value_flag	[1,0,0,1,0,1,0,1,1,0]	为了计算每个row index对应的sample id 个数。就是找到哪些点是新行起始点
hash_value_flag_sumed	[1,1,1,2,2,3,3,4,5,5]	对 new_hash_value_flag 合并，目的是得到每个group（row index）内部包含多少元素。
hash_value_index_count_offset	[0, 3, 5, 7, 8, 0, 0, 0, 0, 0]	每个block要处理低维稠密矩阵一行。如 bid = 1，它希望更新低维稠密矩阵第2行，但想知道更新几次。所以先从 hash_value_index_count_offset[1] 得到了数值 3，然后找到 hash_value_index_sort[3]。因为embedding_feature 最多只有5行（nnz个数），所以这里取前五个即可

最终思路如下:
      每个block要处理低维稠密矩阵一行。假如bid=0 想更新低维矩阵第一行，就是要更新10对应的低维矩阵稠密向量。
      bid对应了key（的梯度），比如 40,50,10,20,30,50,10,30,20,10 这些，其key就是10～50这个5个。

      hash_value_index_count_offset ：本bid对于低维稠密矩阵该行要更新几次。
       sum_num = hash_value_index_count_offset[1] - hash_value_index_count_offset[0] = 3 - 0 = 3个，所以更新3次。

      hash_value_index_sort ：在 [1,1,1,2,2,3,3,4,5,5] 这里找到 1,1,1，
       表示本batch之中一共有3个key对最终embedding_feature第一行做出了贡献。

      所以 bid = 0 ，就是hash_table_value[0]这一行 有三个1，应该更新3次。
      sample_id_sort ：更新就是累积，就是这3次更新分别去输入梯度哪一行去找？3个10分别在梯度的0,1,3这几行。
* */
      value_count_kernel_2<<<grid_size, block_size, 0, stream>>>(
          nnz, new_hash_value_flag.get_ptr(), hash_value_flag_sumed.get_ptr(),
          hash_value_index_count_offset.get_ptr(), hash_value_index_count_counter.get_ptr());

      uint32_t hash_hash_value_index_count_num = 0;
      // this async memcpy will not perform as a async operation because the host memory is not
      // a pinned memroy
      CK_CUDA_THROW_(cudaMemcpyAsync(&hash_hash_value_index_count_num,
                                     hash_value_index_count_counter.get_ptr(), sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, stream));

      // step5: use optimizer method to compute deltaw and update the parameters
      block_size = embedding_vec_size;
      grid_size = max(1, hash_hash_value_index_count_num);

      switch (opt_params.update_type) {
        case Update_t::Global: {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam: {
              float alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));
              // update target mi and vi
              opt_adam_kernel_global<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  opt_tensor.opt_m_tensors_.get_ptr(), opt_tensor.opt_v_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(), opt_params.scaler);
              // all update according to the mi vi
              adam_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.adam,
                  opt_tensor.opt_m_tensors_.get_ptr(), opt_tensor.opt_v_tensors_.get_ptr(), alpha_t,
                  hash_table_value.get_ptr());
              break;
            }
/*
             6.7 更新权重
             这是最后一步，对应了如下：
             step5: use optimizer method to compute deltaw and update the parameters
             调用代码如下：
              注意，这里传递的是 sample_id_sort [0,1,3,0,2,1,2,0,0,1]，
              对应的 hash_value_index_sort 是 [1,1,1,2,2,3,3,4,5,5]，
              hash_value_index_count_offset 是 [0, 3, 5, 7, 8, 0, 0, 0, 0, 0]。

              很明显可以看到，其就是使用权重更新 hash_table_value

最终具体如下图：
 007-009
至此，我们关于 DistributedSlotSparseEmbeddingHash 分析全部完成，下一篇介绍 LocalSlotSparseEmbeddingHash。
 * */
            case Optimizer_t::AdaGrad: {
              opt_adagrad_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.adagrad, opt_tensor.opt_accm_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            }
            case Optimizer_t::MomentumSGD:
              opt_momentum_sgd_kernel_global<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.momentum, opt_tensor.opt_momentum_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(), opt_params.scaler);
              momentum_sgd_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.momentum,
                  opt_tensor.opt_momentum_tensors_.get_ptr(), hash_table_value.get_ptr());
              break;
            case Optimizer_t::Nesterov:
              nesterov_global_update_kernel_global<<<1024, 256, 0, stream>>>(
                  embedding_vec_size, max_vocabulary_size_per_gpu, opt_params.hyperparams.nesterov,
                  opt_tensor.opt_accm_tensors_.get_ptr(), hash_table_value.get_ptr());
              nesterov_local_update_kernel_global<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, opt_tensor.opt_accm_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            case Optimizer_t::SGD:
              // Note: this is in fact a local update
              /// TODO: remove duplicate?
              opt_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }  // switch (optimizer)
          break;
        }
        case Update_t::Local: {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam: {
              float alpha_t =
                  opt_params.lr *
                  sqrt(1 -
                       pow(opt_params.hyperparams.adam.beta2, opt_params.hyperparams.adam.times)) /
                  (1 - pow(opt_params.hyperparams.adam.beta1, opt_params.hyperparams.adam.times));

              opt_adam_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  opt_tensor.opt_m_tensors_.get_ptr(), opt_tensor.opt_v_tensors_.get_ptr(), alpha_t,
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            }
            case Optimizer_t::AdaGrad: {
              opt_adagrad_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.adagrad, opt_tensor.opt_accm_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            }
            case Optimizer_t::MomentumSGD:
              opt_momentum_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.momentum, opt_tensor.opt_momentum_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            case Optimizer_t::Nesterov:
              opt_nesterov_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  opt_params.hyperparams.nesterov, opt_tensor.opt_accm_tensors_.get_ptr(),
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            case Optimizer_t::SGD:
              opt_sgd_kernel<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.lr,
                  sample_id_sort.get_ptr(), hash_value_index_sort.get_ptr(),
                  hash_value_index_count_offset.get_ptr(), wgrad.get_ptr(),
                  hash_table_value.get_ptr(), opt_params.scaler);
              break;
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }  // switch (optimizer)
          break;
        }
        case Update_t::LazyGlobal: {
          switch (opt_params.optimizer) {
            case Optimizer_t::Adam: {
              const float alpha_t_common =
                  opt_params.lr / (1.0f - opt_params.hyperparams.adam.beta1);

              opt_adam_kernel_lazy<<<grid_size, block_size, 0, stream>>>(
                  hash_hash_value_index_count_num, embedding_vec_size, opt_params.hyperparams.adam,
                  opt_tensor.opt_prev_time_tensors_.get_ptr(), opt_tensor.opt_m_tensors_.get_ptr(),
                  opt_tensor.opt_v_tensors_.get_ptr(), alpha_t_common,
                  opt_params.hyperparams.adam.times, sample_id_sort.get_ptr(),
                  hash_value_index_sort.get_ptr(), hash_value_index_count_offset.get_ptr(),
                  wgrad.get_ptr(), hash_table_value.get_ptr(), opt_params.scaler);
              break;
            }
            case Optimizer_t::AdaGrad:
            case Optimizer_t::MomentumSGD:
            case Optimizer_t::Nesterov:
            case Optimizer_t::SGD: {
              /// TODO: implement lazy global update for other optimizer types
              CK_THROW_(Error_t::WrongInput,
                        "Error: lazy global update is only implemented for Adam");
              break;
            }
            default:
              CK_THROW_(Error_t::WrongInput, "Error: Invalid opitimizer type");
          }
          break;
        }
        default:
          CK_THROW_(Error_t::WrongInput, "Error: Invalid update type");
      }  // switch (update type)
    }
#ifndef NDEBUG
    cudaDeviceSynchronize();
    CK_CUDA_THROW_(cudaGetLastError());
#endif
  } catch (const std::runtime_error &rt_err) {
    std::cerr << rt_err.what() << std::endl;
    throw;
  }

  return;
}

template class EmbeddingOptimizer<unsigned int, float>;
template class EmbeddingOptimizer<long long, float>;
template class EmbeddingOptimizer<unsigned int, __half>;
template class EmbeddingOptimizer<long long, __half>;
}  // namespace HugeCTR
