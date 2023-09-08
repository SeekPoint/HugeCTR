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

#include <assert.h>
#include <cublas_v2.h>
#include <curand.h>
#include <nvml.h>

#include <algorithm>
#include <config.hpp>
#include <ctime>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <base/debug/logger.hpp>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef ENABLE_MPI
#include <limits.h>
#include <mpi.h>
#include <stdint.h>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "no suitable MPI type for size_t"
#endif

#endif

#define PYTORCH_INIT

namespace HugeCTR {

#define HUGECTR_VERSION_MAJOR 3
#define HUGECTR_VERSION_MINOR 2
#define HUGECTR_VERSION_PATCH 1

#define WARP_SIZE 32

namespace hybrid_embedding {

enum class HybridEmbeddingType;
enum class CommunicationType;

}  // namespace hybrid_embedding

enum class Check_t { Sum, None };

enum class DataReaderSparse_t { Distributed, Localized };

enum class DataReaderType_t { Norm, Raw, Parquet, RawAsync };

enum class SourceType_t { FileList, Mmap, Parquet };

enum class TrainPSType_t { Staged, Cached };

struct NameID {
  std::string file_name;
  unsigned int id;
};

/**
 * An internal exception, as a child of std::runtime_error, to carry the error code.
 */
class internal_runtime_error : public std::runtime_error {
 private:
  const Error_t err_;

 public:
  /**
   * Get the error code from exception.
   * @return error
   **/
  Error_t get_error() const { return err_; }
  /**
   * Ctor
   */
  internal_runtime_error(Error_t err, std::string str) : runtime_error(str), err_(err) {}
};

enum class LrPolicy_t { fixed };

enum class Optimizer_t { Adam, AdaGrad, MomentumSGD, Nesterov, SGD };

enum class Update_t { Local, Global, LazyGlobal };

// TODO: Consider to move them into a separate file
enum class Activation_t { Relu, None };

enum class FcPosition_t { None, Head, Body, Tail, Isolated };

enum class Regularizer_t { L1, L2 };

enum class Alignment_t { Auto, None };

/*4.5.2 层实现
HugeCTR 属于一个具体而微的深度学习系统，它实现的具体层类型如下：*/
enum class Layer_t {
  BatchNorm,
  BinaryCrossEntropyLoss,
  Reshape,
  Concat,
  CrossEntropyLoss,
  Dropout,
  ELU,
  InnerProduct,
  FusedInnerProduct,
  Interaction,
  MultiCrossEntropyLoss,
  ReLU,
  ReLUHalf,
  GRU,
  MatrixMultiply,
  Scale,
  FusedReshapeConcat,
  FusedReshapeConcatGeneral,
  Softmax,
  PReLU_Dice,
  ReduceMean,
  Sub,
  Gather,
  Sigmoid,
  Slice,
  WeightMultiply,
  FmOrder2,
  Add,
  ReduceSum,
  MultiCross,
  Cast,
  DotProduct,
  ElementwiseMultiply
};

enum class Embedding_t {
  DistributedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingHash,
  LocalizedSlotSparseEmbeddingOneHot,
  HybridSparseEmbedding
};

enum class Initializer_t { Default, Uniform, XavierNorm, XavierUniform, Zero };

enum class TrainState_t {
  Init,
  BottomMLPFprop,
  TopMLPFprop,
  BottomMLPBprop,
  TopMLPBprop,
  MLPExchangeWgrad,
  MLPUpdate,
  Finalize
};

enum class Distribution_t { Uniform, PowerLaw };

enum class PowerLaw_t { Long, Medium, Short, Specific };

// TODO: Consider to move them into a separate file
struct TrainState {
  TrainState_t state = TrainState_t::Init;
  cudaEvent_t* event = nullptr;
};

struct AsyncParam {
  int num_threads;
  int num_batches_per_thread;
  int io_block_size;
  int io_depth;
  int io_alignment;
  bool shuffle;
  Alignment_t aligned_type;
};

struct HybridEmbeddingParam {
  size_t max_num_frequent_categories;
  int64_t max_num_infrequent_samples;
  double p_dup_max;
  double max_all_reduce_bandwidth;
  double max_all_to_all_bandwidth;
  double efficiency_bandwidth_ratio;
  hybrid_embedding::CommunicationType communication_type;
  hybrid_embedding::HybridEmbeddingType hybrid_embedding_type;
};

/*2.1 Norm
为了最大化数据加载性能并最小化存储，Norm 数据集格式由一组二进制数据文件和一个 ASCII 格式的文件列表组成。
 模型文件应指定训练和测试（评估）集的文件名，样本中的元素（键）最大数目和标签维度，具体如图 1（a）所示。

2.1.1 数据文件
        一个数据文件是一个读取线程的最小读取粒度，因此每个文件列表中至少需要10个文件才能达到最佳性能。数据文件由header和实际表格（tabular ）数据组成。

            Header定义：
*/
typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_sum
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
//  long  long error_check;       //0: 没有错误检查；1：check_num
//  long  long number_of_records； //此数据文件中的样本数
//      long  long label_dim;          //标签的维度
//  long  long density_dim;        //密集特征的维度
//  long  long slot_num;           //每个嵌入的 slot_num
//  long  long reserved[ 3 ];      //保留以备将来使用
} DataSetHeader;

#define DISALLOW_COPY(ClassName)        \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete;

#define DISALLOW_MOVE(ClassName)   \
  ClassName(ClassName&&) = delete; \
  ClassName& operator=(ClassName&&) = delete;

#define DISALLOW_COPY_AND_MOVE(ClassName) \
  DISALLOW_COPY(ClassName)                \
  DISALLOW_MOVE(ClassName)

#ifdef ENABLE_MPI
#define CK_MPI_THROW_(cmd)                                                                       \
  do {                                                                                           \
    auto retval = (cmd);                                                                         \
    if (retval != MPI_SUCCESS) {                                                                 \
      throw internal_runtime_error(                                                              \
          Error_t::MpiError, std::string("MPI Runtime error: ") + std::to_string(retval) + " " + \
                                 __FILE__ + ":" + std::to_string(__LINE__) + " \n");             \
    }                                                                                            \
  } while (0)

#endif

#define CK_(err)                                                                       \
  do {                                                                                 \
    Error_t retval = (err);                                                            \
    if (retval != Error_t::Success) {                                                  \
      std::cerr << "[HCDEBUG][ERROR] Return Error: at " << __FILE__ << ":" << __LINE__ \
                << std::endl;                                                          \
    }                                                                                  \
  } while (0)

inline void ERROR_MESSAGE_(const std::string msg) {
  std::string str = msg;
  str += std::string(" ") + __FILE__ + ":" + std::to_string(__LINE__);
  HugeCTR::Logger::get().log(-1, true, true, "%s", str.c_str());
}

#define CK_THROW_(x, msg)                                                                       \
  do {                                                                                          \
    Error_t retval = (x);                                                                       \
    if (retval != Error_t::Success) {                                                           \
      throw internal_runtime_error(x, std::string("Runtime error: ") + (msg) + " " + __FILE__ + \
                                          ":" + std::to_string(__LINE__) + " \n");              \
    }                                                                                           \
  } while (0)

#define CK_RETURN_(x, msg)                                                         \
  do {                                                                             \
    Error_t retval = (x);                                                          \
    if (retval != Error_t::Success) {                                              \
      std::cerr << std::string("Runtime error: ") + (msg) + " " + __FILE__ + ":" + \
                       std::to_string(__LINE__) + " \n";                           \
      return x;                                                                    \
    }                                                                              \
  } while (0)

inline void MESSAGE_(const std::string msg, bool per_process = false, bool new_line = true,
                     bool timestamp = true) {
  std::string final_msg = msg;
  if (new_line) {
    final_msg += "\n";
  }
  Logger::get().log(LOG_INFO_LEVEL, per_process, timestamp, "%s", final_msg.c_str());
}

#define CK_CUDA_THROW_(x)                                                                          \
  do {                                                                                             \
    cudaError_t retval = (x);                                                                      \
    if (retval != cudaSuccess) {                                                                   \
      throw internal_runtime_error(Error_t::CudaError,                                             \
                                   std::string("Runtime error: ") + (cudaGetErrorString(retval)) + \
                                       " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");   \
    }                                                                                              \
  } while (0)

#define CK_NVML_THROW_(x)                                                                          \
  do {                                                                                             \
    nvmlReturn_t retval = (x);                                                                     \
    if (retval != NVML_SUCCESS) {                                                                  \
      throw HugeCTR::internal_runtime_error(HugeCTR::Error_t::NvmlError,                           \
                                            std::string("Runtime error: ") +                       \
                                                (nvmlErrorString(retval)) + " " + __FILE__ + ":" + \
                                                std::to_string(__LINE__) + " \n");                 \
    }                                                                                              \
  } while (0)

#define CK_CUDA_RETURN_BOOL_(x)  \
  do {                           \
    cudaError_t retval = (x);    \
    if (retval != cudaSuccess) { \
      return false;              \
    }                            \
  } while (0)

#ifdef ENABLE_MPI
#define PRINT_FUNC_NAME_()                                                            \
  do {                                                                                \
    int __PID(-1), __NUM_PROCS(-1);                                                   \
    MPI_Comm_rank(MPI_COMM_WORLD, &__PID);                                            \
    MPI_Comm_size(MPI_COMM_WORLD, &__NUM_PROCS);                                      \
    std::cout << "[HCDEBUG][CALL] " << __FUNCTION__ << " in pid: " << __PID << " of " \
              << __NUM_PROCS << " processes." << std::endl;                           \
  } while (0)
#else
#define PRINT_FUNC_NAME_()                                               \
  do {                                                                   \
    std::cout << "[HCDEBUG][CALL] " << __FUNCTION__ << " " << std::endl; \
  } while (0)
#endif

#define CK_CU_RESULT_(x)                                                                        \
  do {                                                                                          \
    if (x > 0) {                                                                                \
      throw internal_runtime_error(                                                             \
          Error_t::CudaError, std::string("CUresult Error, error code: ") + std::to_string(x) + \
                                  ", " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");    \
    }                                                                                           \
  } while (0)

#define CK_CUBLAS_THROW_(x)                                                                        \
  do {                                                                                             \
    cublasStatus_t retval = (x);                                                                   \
    if (retval == CUBLAS_STATUS_NOT_INITIALIZED) {                                                 \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_not_initialized ") +  \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_ARCH_MISMATCH) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_arch_mismatch ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_NOT_SUPPORTED) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_not_supported ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_INVALID_VALUE) {                                                   \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_invalid_value ") +    \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
    if (retval == CUBLAS_STATUS_EXECUTION_FAILED) {                                                \
      throw internal_runtime_error(Error_t::CublasError, std::string("Runtime error: ") +          \
                                                             ("cublas_status_execution_failed ") + \
                                                             __FILE__ + ":" +                      \
                                                             std::to_string(__LINE__) + " \n");    \
    }                                                                                              \
  } while (0)

#define CK_NCCL_THROW_(cmd)                                                                        \
  do {                                                                                             \
    ncclResult_t r = (cmd);                                                                        \
    if (r != ncclSuccess) {                                                                        \
      throw internal_runtime_error(Error_t::NcclError, std::string("Runtime error: NCCL Error ") + \
                                                           std::string(ncclGetErrorString(r)) +    \
                                                           " " + __FILE__ + ":" +                  \
                                                           std::to_string(__LINE__) + " \n");      \
    }                                                                                              \
  } while (0)

#define CK_CUDNN_THROW_(cmd)                                                                      \
  do {                                                                                            \
    cudnnStatus_t retval = (cmd);                                                                 \
    if (retval != CUDNN_STATUS_SUCCESS) {                                                         \
      throw internal_runtime_error(                                                               \
          Error_t::CudnnError, std::string("CUDNN Runtime error: ") +                             \
                                   std::string(cudnnGetErrorString(cmd)) + " " + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " \n");                             \
    }                                                                                             \
  } while (0)

#define CK_CURAND_THROW_(cmd)                                                                    \
  do {                                                                                           \
    curandStatus_t retval = (cmd);                                                               \
    if (retval != CURAND_STATUS_SUCCESS) {                                                       \
      throw internal_runtime_error(                                                              \
          Error_t::CurandError, std::string("CURAND Runtime error: ") + std::to_string(retval) + \
                                    " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");    \
    }                                                                                            \
  } while (0)

template <typename T>
inline void print_func(T& t) {
  std::cout << t << ", ";
  return;
}

template <typename... Args>
inline void LOG(const Args&... args) {
  std::cout << "[";
  std::initializer_list<char>{(print_func(args), 'a')...};
  std::cout << "]" << std::endl;

  return;
}

/*
 DataReaderSparseParam 是依据配置得到的Sparse参数的元信息，其主要成员变量如下：

    sparse_name是其后续层引用的稀疏输入张量的名称。没有默认值，应由用户指定。

    nnz_per_slot是每个插槽的指定sparse输入的最大特征数。
       'nnz_per_slot'可以是'int'，即每个slot的平均nnz，因此每个实例的最大功能数应该是'nnz_per_slot*slot_num'。
       或者可以使用List[int]初始化'nnz_per_slot'，则每个样本的最大特征数应为'sum(nnz_per_slot)'，在这种情况下，数组'nnz_per_slot'的长度应与'slot_num'相同。
   'is_fixed_length'用于标识所有样本中每个插槽的categorical inputs是否具有相同的长度。

   如果不同的样本对于每个插槽具有相同数量的特征，则用户可以设置“is_fixed_length=True”，Hugetr可以使用此信息来减少数据传输时间。

   slot_num指定用于数据集中此稀疏输入的插槽数。
        注意：如果指定了多个'DataReaderSparseParam'，则任何一对'DataReaderSparseParam'之间都不应有重叠。
        比如，在[wdl样本]（../samples/wdl/wdl.py）中，我们总共有27个插槽；我们将第一个插槽指定为"wide_data"，将接下来的26个插槽指定为"deep_data"。

 之前提到了Parser是解析配置文件，HugeCTR 也支持代码设置，参考/搜索  [hugectr.DataReaderSparseParam("wide_data", 30, True, 1)

5.1.1 DataReaderSparseParam
DataReaderSparseParam 定义如下，其中slot_num就代表了这个层拥有几个slot。比如 hugectr.DataReaderSparseParam("deep_data", 2, False, 26) 就代表了有26个slots。

5.1.2 slot概念
我们从文档之中可以知道，slot的概念就是特征域或者表。
In HugeCTR, a slot is a feature field or table. The features in a slot can be one-hot or multi-hot.
The number of features in different slots can be various. You can specify the number of slots (`slot_num`) in the data layer of your configuration file.
Field 或者 slots（有的文章也称其为Feature Group）就是若干有关联特征的集合，其主要作用就是把相关特征组成一个feature field，
然后把这个field再转换为一个稠密向量，这样就可以减少DNN输入层规模和模型参数。

比如：用户看过的商品，用户购买的商品，这就是两个Field，具体每个商品则是feature，这些商品共享一个商品列表，或者说共享一份Vocabulary。
 如果把每个商品都进行embedding，然后把这些张量拼接起来，那么DNN输入层就太大了。
 所以把这些购买的商品归类为同一个field，把这些商品的embedding向量再做pooling之后得到一个field的向量，那么输入层数目就少了很多。

5.1.2.1 FLEN
FLEN: Leveraging Field for Scalable CTR Prediction 之中有一些论证和几张精彩图例可以来辅助说明这个概念，具体如下：

CTR预测任务中的数据是多域（multi-field categorical ）的分类数据，也就是说，每个特征都是分类的，并且只属于一个字段。
 例如，特征 "gender=Female "属于域 "gender"，特征 "age=24 "属于域 "age"，特征 "item category=cosmetics "属于域 "item category"。
 特征 "性别 "的值是 "男性 "或 "女性"。特征 "年龄 "被划分为几个年龄组。"0-18岁"，"18-25岁"，"25-30岁"，等等。
 人们普遍认为，特征连接（conjunctions）是准确预测点击率的关键。
 一个有信息量的特征连接的例子是：年龄组 "18-25 "与性别 "女性 "相结合，用于 "化妆品 "项目类别。
 它表明，年轻女孩更有可能点击化妆品产品。
004-clip-03.png
FLEN模型中使用了一个filed-wise embedding vector，通过将同一个域（如user filed或者item field）之中的embedding 向量进行求和，来得到域对应的embedding向量。

比如，首先把特征xn转换为嵌入向量 en
en=Vnxn
其次，使用 sum-pooling 得到 field-wise embedding vectors。
em=∑n|F(n)=men
比如 004-012.png
最后，把所有field-wise embedding vectors拼接起来。
 004-013.png
系统整体架构如下：
004-014.png
 */
struct DataReaderSparseParam {
  std::string top_name;
  std::vector<int> nnz_per_slot;
  bool is_fixed_length;
  int slot_num;

  DataReaderSparse_t type;
  int max_feature_num;
  int max_nnz;

  DataReaderSparseParam() {}
  DataReaderSparseParam(const std::string& top_name_, const std::vector<int>& nnz_per_slot_,
                        bool is_fixed_length_, int slot_num_)
      : top_name(top_name_),
        nnz_per_slot(nnz_per_slot_),
        is_fixed_length(is_fixed_length_),
        slot_num(slot_num_),
        type(DataReaderSparse_t::Distributed) {
    if (static_cast<size_t>(slot_num_) != nnz_per_slot_.size()) {
      CK_THROW_(Error_t::WrongInput, "slot num != nnz_per_slot.size().");
    }
    max_feature_num = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);
    max_nnz = *std::max_element(nnz_per_slot.begin(), nnz_per_slot.end());
  }

  DataReaderSparseParam(const std::string& top_name_, const int nnz_per_slot_,
                        bool is_fixed_length_, int slot_num_)
      : top_name(top_name_),
        nnz_per_slot(slot_num_, nnz_per_slot_),
        is_fixed_length(is_fixed_length_),
        slot_num(slot_num_),
        type(DataReaderSparse_t::Distributed) {
    max_feature_num = std::accumulate(nnz_per_slot.begin(), nnz_per_slot.end(), 0);
    max_nnz = *std::max_element(nnz_per_slot.begin(), nnz_per_slot.end());
  }
};

}  // namespace HugeCTR

#include <profiler.hpp>
