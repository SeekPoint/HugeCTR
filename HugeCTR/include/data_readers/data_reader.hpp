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

#include <atomic>
#include <common.hpp>
#include <data_reader.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/data_collector.hpp>
#include <data_readers/data_reader_common.hpp>
#include <data_readers/data_reader_worker_group.hpp>
#include <data_readers/data_reader_worker_group_norm.hpp>
#include <data_readers/data_reader_worker_group_parquet.hpp>
#include <data_readers/data_reader_worker_group_raw.hpp>
#include <filesystem>
#include <fstream>
#include <gpu_resource.hpp>
#include <tensor2.hpp>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {




//0x03 DataReader Buffer 机制
//我们接下来看看 DataReader 的若干Buffer，依赖于这些buffer，HugeCTR实现了流水线的前两级。
//
//3.1 比对
//   我们首先要做一个历史对比，看看这部分代码的发展脉络。我们先看看3.1版本的代码。DataReader 我们选取了部分成员变量。3.1 版本之前使用了一个heap进行操作，即下面的csr_heap_。
//class DataReader : public IDataReader {
// std::shared_ptr<HeapEx<CSRChunk<TypeKey>>> csr_heap_; /**< heap to cache the data set */
// Tensors2<float> label_tensors_;                       /**< Label tensors for the usage of loss */
// std::vector<TensorBag2> dense_tensors_;               /**< Dense tensors for the usage of loss */
// /* Each gpu will have several csr output for different embedding */
// Tensors2<TypeKey> csr_buffers_; /**< csr_buffers contains row_offset_tensor and value_tensors */
// Tensors2<TypeKey> row_offsets_tensors_; /**< row offset tensors*/
// Tensors2<TypeKey> value_tensors_;       /**< value tensors */
// std::vector<std::shared_ptr<size_t>> nnz_array_;
//
// const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
// const size_t dense_dim_; /**< dimention of dense */
//}
//我们再看看3.2.1版本的代码，也选取了部分成员变量。
//
//template <typename TypeKey>
//class DataReader : public IDataReader {
// std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;  // gpu_id -> thread_idx
// std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
// std::shared_ptr<DataReaderOutput> output_;
//
// const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
// const size_t dense_dim_; /**< dimention of dense */
//}
//3.2.1 这里是：
//
//   把 label_tensors_, dense_tensors_ 移动到 AsyncReader。
//       把 csr_heap_ 用 thread_buffers_，broadcast_buffer_，output_ 等进行替代。
//       把 row_offsets_tensors_，value_tensors_，nnz_array_ 等等用 ThreadBuffer，BroadcastBuffer，DataReaderOutput 之中的 SparseTensorBag 来包括，统一管理 CSR。
//   3.2 Buffer 相关类
//       我们依据上面的历史版本比对来看看。
//           在之前版本（比如3.1）之中，存在一个 HeapEX 类，其实现了 CPU 到 GPU 之间的一个数据缓存功能。
//           在最新版本之中，改为一系列 buffer 相关类，比如 ThreadBuffer 和 BroadcastBuffer，其状态都是由 BufferState 实现的。
//   enum class BufferState : int { FileEOF, Reading, ReadyForRead, Writing, ReadyForWrite };

/**
 * @brief Data reading controller.
 *
 * Control the data reading from data set to embedding.
 * An instance of DataReader will maintain independent
 * threads for data reading (IDataReaderWorker)
 * from dataset to heap. Meanwhile one independent
 * thread consumes the data (DataCollector),
 * and copy the data to GPU buffer.
 */
template <typename TypeKey>
class DataReader : public IDataReader {
 private:
  //从静态角度看，主要是以下三个buffer：
  std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;  // gpu_id -> thread_idx //线程内部使用的buffer。
  std::shared_ptr<BroadcastBuffer> broadcast_buffer_; //用来后续和collector交互，collector 把它作为中间buffer。
  std::shared_ptr<DataReaderOutput> output_;    //reader的输出，训练最后读取的是这里。


/*
   3.3.2 ThreadBuffer
 然后我们看看处理 thread_buffers_ 部分，这里是为线程buffer进行处理。我们首先获取ThreadBuffer类定义如下，后面分析时候可以比对。

 struct ThreadBuffer {
   std::vector<SparseTensorBag> device_sparse_buffers;  // same number as embedding number
   std::vector<unsigned char> is_fixed_length;          // same number as embedding number
   TensorBag2 device_dense_buffers;
   std::atomic<BufferState> state;
   long long current_batch_size;
   int batch_size;
   size_t param_num;
   int label_dim;
   int dense_dim;
   int batch_size_start_idx;  // dense buffer
   int batch_size_end_idx;
 };
 其次，具体构建函数中的逻辑如下：
     首先，对于 thread_buffers_ 这个vector，会拓展 vector 容量到线程数大小。
     拿到本线程（或者说是本GPU）在buffs之中对应的buffer，赋值到 buff。
     对于每一个线程，会生成一个ThreadBuffer，命名为current_thread_buffer，放入到 thread_buffers_ 之中。
     对于每一个 ThreadBuffer，预留 ThreadBuffer 的device_sparse_buffers 和 is_fixed_length 这两个 vector 的容量大小。
     遍历sparse参数，对于每一个参数，会建立一个临时张量，并且通过 buff 预留内存（CPU或者GPU），然后把此临时张量放入device_sparse_buffers。
     建立一个针对dense的张量，并且通过 buff 预留张量内存，把临时张量放入device_dense_buffers。
     设置current_thread_buffer 状态。
     设置 current_thread_buffer 其他信息。
*/

  /*
从动态角度看，成员变量之中重要的是以下两个：
worker_group ：工作线程组，负责把数据从dataset文件读取到内存之中，这个可以认为是流水线的第一级。
   之前的版本之中有一个HeapEx数据结构用来做中间缓存，目前这个数据结构已经移除。

data_collector_ ：拥有一个线程，负责把数据拷贝到GPU之中。
   这个可以认为是流水线的第二级。
   */
  std::shared_ptr<DataReaderWorkerGroup> worker_group_;
  std::shared_ptr<DataCollector<TypeKey>> data_collector_; /**< pointer of DataCollector */

  /* Each gpu will have several csr output for different embedding */
  const std::vector<DataReaderSparseParam> params_;
  std::shared_ptr<ResourceManager> resource_manager_; /**< gpu resource used in this data reader*/
  const size_t batchsize_;                            /**< batch size */
  const size_t label_dim_; /**< dimention of label e.g. 1 for BinaryCrossEntropy */
  const size_t dense_dim_; /**< dimention of dense */
  long long current_batchsize_;

  bool repeat_;
  std::string file_name_;
  SourceType_t source_type_;

 public:
  /*
  对DataReader的构建分为两部分：
  在构造函数之中会：
      对各种buffer进行配置。
      对构建DataCollector。
  在create_datareader之中会分别处理 train_data_reader和 evaluate_data_reader，也就是用于训练和评估的两个reader。然后会为他们建立workgroup。
  */
  //其主要功能就是为三种buffer来预留空间，分配内存，最后构建了collector
  DataReader(int batchsize, size_t label_dim, int dense_dim,
             std::vector<DataReaderSparseParam> &params,
             const std::shared_ptr<ResourceManager> &resource_manager, bool repeat, int num_threads,
             bool use_mixed_precision)
      : broadcast_buffer_(new BroadcastBuffer()),
        output_(new DataReaderOutput()),
        params_(params),
        resource_manager_(resource_manager),
        batchsize_(batchsize),
        label_dim_(label_dim),
        dense_dim_(dense_dim),
        repeat_(repeat) {
    size_t local_gpu_count = resource_manager_->get_local_gpu_count();
    size_t total_gpu_count = resource_manager_->get_global_gpu_count();

    // input check
    if (total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 ||
        0 != batchsize_ % total_gpu_count) {
      CK_THROW_(Error_t::WrongInput,
                "total_gpu_count == 0 || batchsize <= 0 || label_dim <= 0 || dense_dim < 0 || 0 != "
                "batchsize_ % total_gpu_count");
    }
    // batchsize_ is a multiple of total_gpu_count
    size_t batch_size_per_gpu = batchsize_ / total_gpu_count;
    // 1. 生成了一个临时变量buffs，用来具体分配内存，里面是若干 CudaAllocator，每个CudaAllocator对应了i个GPU
    std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> buffs;

    // 先预留部分内存空间
    buffs.reserve(local_gpu_count);

    // 为每个GPU初始化一个GeneralBuffer2
    for (size_t i = 0; i < local_gpu_count; ++i) {
      buffs.push_back(GeneralBuffer2<CudaAllocator>::create());
    }

    // 2.预留buffer
    // 处理 thread_buffers_，会拓展 vector 容量到线程数大小
    thread_buffers_.reserve(num_threads);
    // 遍历线程
    for (int i = 0; i < num_threads; ++i) {
      // a worker may maintain multiple buffers on device i % local_gpu_count
      auto local_gpu = resource_manager_->get_local_gpu(i % local_gpu_count);
      CudaCPUDeviceContext context(local_gpu->get_device_id());
      // 找到对应GPU对应的CudaAllocator，进行分配
      // 拿到本线程（或者说是本GPU）在buffs之中对应的buffer
      auto &buff = buffs[i % local_gpu_count];

      // 生成一个ThreadBuffer，存入到thread_buffers_
      std::shared_ptr<ThreadBuffer> current_thread_buffer = std::make_shared<ThreadBuffer>();
      thread_buffers_.push_back(current_thread_buffer);

      // 预留 ThreadBuffer 的device_sparse_buffers 和 is_fixed_length 这两个 vector 的容量大小
      current_thread_buffer->device_sparse_buffers.reserve(params.size());
      current_thread_buffer->is_fixed_length.reserve(params.size());  // vector的reserve

      // 遍历参数
      for (size_t param_id = 0; param_id < params.size(); ++param_id) {
        auto &param = params_[param_id];
        SparseTensor<TypeKey> temp_sparse_tensor;
        // 预留内存
        // 建立一个临时张量，并且预留内存（CPU或者GPU）
        buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                      &temp_sparse_tensor);
        // 把张量放入device_sparse_buffers
        current_thread_buffer->device_sparse_buffers.push_back(temp_sparse_tensor.shrink());
        current_thread_buffer->is_fixed_length.push_back(param.is_fixed_length);
      }
      // 建立一个针对dense的张量
      Tensor2<float> temp_dense_tensor;

      // 预留内存  // 预留张量内存
      buff->reserve({batch_size_per_gpu * local_gpu_count, label_dim + dense_dim},
                    &temp_dense_tensor);
      // 把临时张量放入device_dense_buffers
      current_thread_buffer->device_dense_buffers = temp_dense_tensor.shrink();
      // 设置状态
      current_thread_buffer->state.store(BufferState::ReadyForWrite);
      // 设置其他信息
      current_thread_buffer->current_batch_size = 0;
      current_thread_buffer->batch_size = batchsize;
      current_thread_buffer->param_num = params.size();
      current_thread_buffer->label_dim = label_dim;
      current_thread_buffer->dense_dim = dense_dim;
      current_thread_buffer->batch_size_start_idx =
          batch_size_per_gpu * resource_manager_->get_gpu_global_id_from_local_id(0);
      current_thread_buffer->batch_size_end_idx =
          current_thread_buffer->batch_size_start_idx + batch_size_per_gpu * local_gpu_count;
    }

    //此时如下，注意，DataReader 包括多个 ThreadBuffer。
    // 003-001.jpg

    // 处理 broadcast buffer，注意这里的reserve是 vector数据结构的方法，不是预留内存
    //按照构建代码来说，这里只是做了一些预留和设置，没有涉及内存，内存在后续会统一处理
    broadcast_buffer_->sparse_buffers.reserve(local_gpu_count * params.size()); // 预留vector的容量
    broadcast_buffer_->is_fixed_length.reserve(local_gpu_count * params.size());// 预留vector的容量
    broadcast_buffer_->dense_tensors.reserve(local_gpu_count); // 预留vector的容量
    broadcast_buffer_->finish_broadcast_events.resize(local_gpu_count);
    // 设置状态
    broadcast_buffer_->state.store(BufferState::ReadyForWrite);
    broadcast_buffer_->current_batch_size = 0;
    broadcast_buffer_->param_num = params.size();

    // 处理 output buffer，注意这里的reserve是 vector数据结构的方法，不是预留内存
    //按照构建代码来说，这里只是做了一些预留和设置，没有涉及内存，内存在后续会统一处理。
    output_->dense_tensors.reserve(local_gpu_count); // 预留vector的容量
    output_->label_tensors.reserve(local_gpu_count); // 预留vector的容量
    output_->use_mixed_precision = use_mixed_precision;
    output_->label_dense_dim = label_dim + dense_dim;
    // 预留sparse tensor，注意这里的reserve是 vector数据结构的方法，不是预留内存
    for (size_t param_id = 0; param_id < params.size(); ++param_id) {
      auto &param = params_[param_id];

      output_->sparse_tensors_map[param.top_name].reserve(local_gpu_count);
      output_->sparse_name_vec.push_back(param.top_name);
    }

    // 3.3.5 预留和分配
    // 这里会对 broadcast 和 output 进行预留，这里统一分配内存
    // 遍历本地的 GPU
    for (size_t local_id = 0; local_id < local_gpu_count; ++local_id) {
      // 还是需要针对每一个GPU，找到对应的CudaAllocator进行分配
      auto local_gpu = resource_manager_->get_local_gpu(local_id);
      CudaDeviceContext ctx(local_gpu->get_device_id());
      // 获取临时buffs之中对应某一个本地gpu的allocator
      auto &buff = buffs[local_id];

      for (size_t param_id = 0; param_id < params.size(); ++param_id) {
        auto &param = params_[param_id];
        SparseTensor<TypeKey> temp_sparse_tensor;
        // 给broadcast_buffer_分配内存  // 分配sparse内存
        buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                      &temp_sparse_tensor);
        // 赋值到broadcast 之上
        broadcast_buffer_->sparse_buffers.push_back(temp_sparse_tensor.shrink());
        broadcast_buffer_->is_fixed_length.push_back(param.is_fixed_length);
      }
      // 分配dense内存
      Tensor2<float> temp_dense_tensor;
      buff->reserve({batch_size_per_gpu, label_dim + dense_dim}, &temp_dense_tensor);
      // 赋值到broadcast 之上
      broadcast_buffer_->dense_tensors.push_back(temp_dense_tensor.shrink());

      CK_CUDA_THROW_(cudaEventCreateWithFlags(&broadcast_buffer_->finish_broadcast_events[local_id],
                                              cudaEventDisableTiming));

      for (size_t param_id = 0; param_id < params.size(); ++param_id) {
        auto &param = params_[param_id];
        // 分配sparse内存
        SparseTensor<TypeKey> temp_sparse_tensor;
        // 预留内存
        buff->reserve({(size_t)batchsize, (size_t)param.max_feature_num}, param.slot_num,
                      &temp_sparse_tensor);
        // 赋值到output之上
        output_->sparse_tensors_map[param.top_name].push_back(temp_sparse_tensor.shrink());
      }

      // 分配label的内存
      Tensor2<float> label_tensor;
      // 预留内存
      buff->reserve({batch_size_per_gpu, label_dim}, &label_tensor);
      // 赋值到output之上
      output_->label_tensors.push_back(label_tensor.shrink());

      if (use_mixed_precision) {
        Tensor2<__half> dense_tensor;
        // 预留内存  // 分配dense内存
        buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
        // 赋值到output之上
        output_->dense_tensors.push_back(dense_tensor.shrink());
      } else {
        Tensor2<float> dense_tensor;
        // 预留内存  // 分配dense内存
        buff->reserve({(size_t)batch_size_per_gpu, (size_t)dense_dim}, &dense_tensor);
        // 赋值到output之上
        output_->dense_tensors.push_back(dense_tensor.shrink());
      }

      // 3. 分配内存
      buff->allocate();  // 统一分配
    }

    /*预留buffer的具体逻辑如下：
003-002.jpg
分配之后如下，需要注意的是，这里都是简化版本，没有体现出来多个本地GPU的状态。
比如下面三个类的成员变量都会分配到多个本地GPU之上。

// embedding number 指的是本模型之中，DataReaderSparseParam 的个数，就是有几个 embedding 层
struct ThreadBuffer {
    std::vector<SparseTensorBag> device_sparse_buffers;  // same number as embedding number
    // device_sparse_buffers 会分配在多个本地GPU之上

struct BroadcastBuffer {
    std::vector<SparseTensorBag>
        sparse_buffers;  // same number as (embedding number * local device number)
    // sparse_buffers 也会分配在多个本地GPU之上

struct DataReaderOutput {
    std::map<std::string, std::vector<SparseTensorBag>> sparse_tensors_map;
    // 每个 sparse_tensors_map[param.top_name] 都会分配在多个本地GPU之上
    // 比如 output_->sparse_tensors_map[param.top_name].reserve(local_gpu_count);
如下简化版本之中都只体现了一个GPU，这些buffer都是位于GPU之上。
现在 DataReader 有了一系列buffer，我们接下来看看如何使用。
  */

    // 4. 构建DataCollector
    data_collector_ = std::make_shared<DataCollector<TypeKey>>(thread_buffers_, broadcast_buffer_,
                                                               output_, resource_manager);
    return;
  }

  ~DataReader() override {
    try {
      // stop all the loops
      data_collector_->stop();
      worker_group_->end();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();
      for (size_t i = 0; i < local_gpu_count; ++i) {
        CK_CUDA_THROW_(cudaEventDestroy(broadcast_buffer_->finish_broadcast_events[i]));
      }
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }

  /**
   * Reading a batch from cpu to gpu (embedding)
   */
  TensorScalarType get_scalar_type() const override {
    return TensorScalarTypeFunc<TypeKey>::get_type();
  }

  long long read_a_batch_to_device() override {
    current_batchsize_ = read_a_batch_to_device_delay_release();
    ready_to_collect();
    return current_batchsize_;
  }  // read data from csr to tensors

//  5.3.2 read_a_batch_to_device_delay_release
//      read_a_batch_to_device_delay_release 是最终配置好embedding数据的地方。
  long long read_a_batch_to_device_delay_release() override {
    current_batchsize_ = data_collector_->read_a_batch_to_device();
    return current_batchsize_;
  }

  void ready_to_collect() override { data_collector_->finalize_batch(); }

  long long get_current_batchsize_per_device(size_t local_id) override {
    if (batchsize_ % resource_manager_->get_global_gpu_count() != 0) {
      CK_THROW_(Error_t::UnspecificError,
                "batchsize_ % resource_manager_->get_global_gpu_count() != 0");
    }
    long long batchsize_per_device = batchsize_ / resource_manager_->get_global_gpu_count();
    size_t global_id = resource_manager_->get_gpu_global_id_from_local_id(local_id);
    long long remain_samples = current_batchsize_ - global_id * batchsize_per_device;
    if (remain_samples >= batchsize_per_device) {
      return batchsize_per_device;
    } else if (remain_samples > 0) {
      return remain_samples;
    } else {
      return 0;
    }
  }

  long long get_full_batchsize() const override { return batchsize_; }

  bool is_started() const override { return worker_group_->is_started(); }

  void start() override { worker_group_->start(); }

  const std::vector<SparseTensorBag> &get_sparse_tensors(const std::string &name) {
    if (output_->sparse_tensors_map.find(name) == output_->sparse_tensors_map.end()) {
      CK_THROW_(Error_t::IllegalCall, "no such sparse output in data reader:" + name);
    }
    return output_->sparse_tensors_map[name];
  }

  const std::vector<TensorBag2> &get_label_tensors() const { return output_->label_tensors; }

  const std::vector<TensorBag2> &get_dense_tensors() const { return output_->dense_tensors; }

  //我们以 norm 为例进行解析，首先提一下，其内部建立了 WorkerGroup。
  /*
我们用create_drwg_norm来继续分析，发现其构建了DataReaderWorkerGroupNorm。
即，配置了 DataReader 之中的成员变量 worker_group_ 为一个 DataReaderWorkerGroupNorm。
注意，这里传入的是thread_buffers_，说明 DataReaderWorkerGroup 操作的就是DataReader 的 thread_buffers_。
   */
  void create_drwg_norm(std::string file_name, Check_t check_type,
                        bool start_reading_from_beginning = true) override {
    source_type_ = SourceType_t::FileList;
    worker_group_.reset(new DataReaderWorkerGroupNorm<TypeKey>(
        thread_buffers_, resource_manager_, file_name, repeat_, check_type, params_,
        start_reading_from_beginning));
    file_name_ = file_name;
  }

  void create_drwg_raw(std::string file_name, long long num_samples, bool float_label_dense,
                       bool data_shuffle = false,
                       bool start_reading_from_beginning = true) override {
    // check if key type compatible with dataset
    size_t file_size = std::filesystem::file_size(file_name);
    size_t expected_file_size = (label_dim_ + dense_dim_) * sizeof(float);
    for (auto &param : params_) {
      expected_file_size += param.slot_num * sizeof(TypeKey);
    }
    expected_file_size *= num_samples;
    if (file_size != expected_file_size) {
      CK_THROW_(Error_t::UnspecificError, "dataset key type and dataset size is not compatible.");
    }
    source_type_ = SourceType_t::Mmap;
    worker_group_.reset(new DataReaderWorkerGroupRaw<TypeKey>(
        thread_buffers_, resource_manager_, file_name, num_samples, repeat_, params_, label_dim_,
        dense_dim_, batchsize_, float_label_dense, data_shuffle, start_reading_from_beginning));
    file_name_ = file_name;
  }

  void create_drwg_parquet(std::string file_name, const std::vector<long long> slot_offset,
                           bool start_reading_from_beginning = true) override {
    source_type_ = SourceType_t::Parquet;
    // worker_group_.empty
    worker_group_.reset(new DataReaderWorkerGroupParquet<TypeKey>(
        thread_buffers_, file_name, repeat_, params_, slot_offset, resource_manager_,
        start_reading_from_beginning));
  }

  void set_source(std::string file_name = std::string()) override {
    if (worker_group_ != nullptr) {
      if (file_name.empty()) {
        if (file_name_.empty()) {
          throw internal_runtime_error(Error_t::NotInitialized, "invalid file_name");
        } else {
          file_name = file_name_;
        }
      }
      worker_group_->set_source(source_type_, file_name, repeat_);
    } else {
      throw internal_runtime_error(Error_t::NotInitialized, "worker_group_ == nullptr");
    }
  }
};
}  // namespace HugeCTR
