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
#include <common.hpp>
#include <data_readers/check_none.hpp>
#include <data_readers/check_sum.hpp>
#include <data_readers/csr.hpp>
#include <data_readers/data_reader_worker_interface.hpp>
#include <data_readers/file_list.hpp>
#include <data_readers/file_source.hpp>
#include <fstream>
#include <vector>

namespace HugeCTR {
/*
其构建代码如下，需要注意，
    有一个继承于基类的变量 std::shared_ptr buffer_ 指向的是 ThreadBuffer。
    变量 host_sparse_buffer_ 是构建在 Host 之上，而非GPU之上，这个 host_sparse_buffer_ 作用是文件中读取数据，解析成csr，放置到 host_sparse_buffer_ 之上。
    关于变量 DataReaderSparseParam 的说明，这是一个DataReaderSparseParam 数组，如果做如下设置，则 params_ 包含三个元素，分别对应分了 user, good, cate。
model.add(hugectr.Input(label_dim = 1, label_name = "label",
                        dense_dim = 0, dense_name = "dense",
                        data_reader_sparse_param_array =
                        [hugectr.DataReaderSparseParam("UserID", 1, True, 1),
                        hugectr.DataReaderSparseParam("GoodID", 1, True, 11),
                        hugectr.DataReaderSparseParam("CateID", 1, True, 11)]))
*/
template <class T>
class DataReaderWorker : public IDataReaderWorker {
 private:
  DataSetHeader
      data_set_header_;  /**< the header of data set, which has main informations of a data file */
  size_t buffer_length_; /**< buffer size for internal use */
  Check_t check_type_;   /**< check type for data set */
  std::vector<DataReaderSparseParam> params_; /**< configuration of data reader sparse input */
  std::shared_ptr<Checker> checker_; /**< checker aim to perform error check of the input data */
  bool skip_read_{false};            /**< set to true when you want to stop the data reading */
  int current_record_index_{0};
  size_t total_slot_num_;
  std::vector<size_t> last_batch_nnz_;

  Tensor2<float> temp_host_dense_buffer_;  // read data to make checker move
  Tensor2<float> host_dense_buffer_;
  std::vector<CSR<T>> host_sparse_buffer_;

  //read_new_file 完成了对文件的读取。
  void read_new_file() {
    constexpr int MAX_TRY = 10;
    for (int i = 0; i < MAX_TRY; i++) {
      if (checker_->next_source() == Error_t::EndOfFile) {
        throw internal_runtime_error(Error_t::EndOfFile, "EndOfFile");
      }

      Error_t err =
          checker_->read(reinterpret_cast<char*>(&data_set_header_), sizeof(DataSetHeader));
      current_record_index_ = 0;
      if (!(data_set_header_.error_check == 0 && check_type_ == Check_t::None) &&
          !(data_set_header_.error_check == 1 && check_type_ == Check_t::Sum)) {
        ERROR_MESSAGE_("DataHeaderError");
        continue;
      }
      if (static_cast<size_t>(data_set_header_.slot_num) != total_slot_num_) {
        ERROR_MESSAGE_("DataHeaderError");
        continue;
      }
      if (err == Error_t::Success) {
        return;
      }
    }
    CK_THROW_(Error_t::BrokenFile, "failed to read a file");
  }

  //Reader构建时候，会建立一个 checker_，用来从文件读取数据。
  void create_checker() {
    switch (check_type_) {
      case Check_t::Sum:
        checker_ = std::make_shared<CheckSum>(*source_);
        break;
      case Check_t::None:
        checker_ = std::make_shared<CheckNone>(*source_);
        break;
      default:
        assert(!"Error: no such Check_t && should never get here!!");
    }
  }

  void post_set_source() override {
    create_checker();

    is_eof_ = false;
    buffer_->state.store(BufferState::ReadyForWrite);
  }

 public:
  /**
   * Ctor
   */
  DataReaderWorker(const int worker_id, const int worker_num,
                   const std::shared_ptr<GPUResource>& gpu_resource, int* loop_flag,
                   const std::shared_ptr<ThreadBuffer>& buffer, const std::string& file_list,
                   size_t buffer_length, bool repeat, Check_t check_type,
                   const std::vector<DataReaderSparseParam>& params)
      : IDataReaderWorker(worker_id, worker_num, gpu_resource, !repeat, loop_flag, buffer),
        buffer_length_(buffer_length),
        check_type_(check_type),
        params_(params),
        total_slot_num_(0),
        last_batch_nnz_(params.size(), 0) {
    if (worker_id >= worker_num) {
      CK_THROW_(Error_t::BrokenFile, "DataReaderWorker: worker_id >= worker_num");
    }
    total_slot_num_ = 0;
    for (auto& p : params) {
      total_slot_num_ += p.slot_num;
    }
    source_ = std::make_shared<FileSource>(worker_id, worker_num, file_list, repeat);
    create_checker();

    int batch_size = buffer->batch_size;
    int batch_size_start_idx = buffer->batch_size_start_idx;
    int batch_size_end_idx = buffer->batch_size_end_idx;
    int label_dim = buffer->label_dim;
    int dense_dim = buffer->dense_dim;

    CudaCPUDeviceContext ctx(gpu_resource->get_device_id());  // 得到了本worker对应哪个GPU
    std::shared_ptr<GeneralBuffer2<CudaHostAllocator>> buff =
        GeneralBuffer2<CudaHostAllocator>::create();

    buff->reserve({static_cast<size_t>(batch_size_end_idx - batch_size_start_idx),
                   static_cast<size_t>(label_dim + dense_dim)},
                  &host_dense_buffer_);
    buff->reserve({static_cast<size_t>(label_dim + dense_dim)}, &temp_host_dense_buffer_);

    for (auto& param : params) {
      host_sparse_buffer_.emplace_back(batch_size * param.slot_num,
                                       batch_size * param.max_feature_num);
    }

    buff->allocate();

    /*
     具体拓展如下，其中每个thread里面含有一个worker：
     003-005.jpg
     或者我们进一步简化几个内存类，得到如下，DataReaderWorker 操作 DataReader 之中的一个 ThreadBuffer，
     003-006.jpg
     */
  }

  /*
   4.7.3 读取批次数据
      read_a_batch 完成具体解析数据集工作。
          首先从文件读取数据。
          等待 ThreadBuffer（就是DataReader的thread_buffers_成员变量）的状态变成ReadyForWrite。
          解析成csr，放入到 host_dense_buffer_。
          调用 wait_until_h2d_ready 等待拷贝完成。
          其次调用cudaMemcpyAsync把数据从 host_dense_buffer_ 拷贝到 ThreadBuffer 之中。这里有两点很重要：
              目前数据在 host_sparse_buffer_（CPU）之上，需要拷贝到 GPU（目标是 ThreadBuffer 的 device_sparse_buffers 成员变量）。
              而且，host_sparse_buffer_ 是 CSR 格式，ThreadBuffer 的 device_sparse_buffers 成员变量是SparseTensor格式，需要转换。
              这里是通过拷贝就进行了转换。
      有几点如下：
          nnz 的意思是：non-zero feature number。
          每一个slot数据对应了一个CSR row。
  */
  /**
   * read a batch of data from data set to heap.
   */
  void read_a_batch() {
    //// 得到各种配置
    long long current_batch_size = buffer_->batch_size;
    int label_dim = buffer_->label_dim;
    int dense_dim = buffer_->dense_dim;
    int label_dense_dim = label_dim + dense_dim;
    int batch_size_start_idx = buffer_->batch_size_start_idx;
    int batch_size_end_idx = buffer_->batch_size_end_idx;

    try {
      if (!checker_->is_open()) {
        read_new_file();  // 读一个新文件
      }
    } catch (const internal_runtime_error& rt_err) {
      Error_t err = rt_err.get_error();
      // TODO: when in repeate mode and the dataset sample num can not devided by batchsize,
      // Norm/Raw have different behavior to last batch. Norm will fetch the data from the begining
      // of the datset, while Raw will output current_batchsize < batchsize. Comment by Alex Liu
      // (2021.7.4)
      if (err == Error_t::EndOfFile) {  // 文件读完了
        if (!wait_until_h2d_ready()) return;  // 等待 buffer_ 状态变为 ReadyForWrite
        buffer_->current_batch_size = 0;
        assert(buffer_->state.load() == BufferState::Writing);   // 设置
        is_eof_ = true;
        buffer_->state.store(BufferState::ReadyForRead);  // 设置状态为可读

        while (buffer_->state.load() != BufferState::ReadyForWrite) {
          usleep(2);
          if (*loop_flag_ == 0) return;  // in case main thread exit
        }
        return;  // need this return to run from begining
      } else {
        throw;
      }
    }

    // if the EOF is faced, the current batch size can be changed later
    if (data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim)
      CK_THROW_(Error_t::WrongInput,
                "data_set_header_.label_dim + data_set_header_.dense_dim != label_dense_dim");

    for (auto& each_csr : host_sparse_buffer_) {
      each_csr.reset();
    }
    // batch loop
    for (int batch_idx = 0; batch_idx < buffer_->batch_size; ++batch_idx) { //读取batch中一个
      if (batch_idx >= current_batch_size) { // 如果已经读取batch之中的全部数据了
        for (size_t param_id = 0; param_id < params_.size(); ++param_id) {  // 多个embedding
          // 如果是前面那个例子，这里遍历的就是user, good, cate
          auto& param = params_[param_id];
          // host_sparse_buffer_类型是std::vector<CSR<T>>
          auto& current_csr = host_sparse_buffer_[param_id];
          for (int k = 0; k < param.slot_num; k++) { // slot数目就是行数
            current_csr.new_row();  // 增加一行
          }
        }
        if (batch_idx >= batch_size_start_idx &&
            batch_idx < batch_size_end_idx) {  // only read local device dense data
          // 设置dense
          float* ptr =
              host_dense_buffer_.get_ptr() + (batch_idx - batch_size_start_idx) * label_dense_dim;

          for (int j = 0; j < label_dense_dim; j++) {
            ptr[j] = 0.f;
          }
        }
        continue;
      }
      try {
        try {
          if (batch_idx >= batch_size_start_idx &&
              batch_idx < batch_size_end_idx) {  // only read local device dense data
            // 读取dense参数
            CK_THROW_(checker_->read(reinterpret_cast<char*>(host_dense_buffer_.get_ptr() +
                                                             (batch_idx - batch_size_start_idx) *
                                                                 label_dense_dim),
                                     sizeof(float) * label_dense_dim),
                      "failure in reading label_dense");
          } else {
            // 读取dense参数
            CK_THROW_(checker_->read(reinterpret_cast<char*>(temp_host_dense_buffer_.get_ptr()),
                                     sizeof(float) * label_dense_dim),
                      "failure in reading label_dense");
          }

          for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
            auto& current_csr = host_sparse_buffer_[param_id];
            current_csr.set_check_point();
          }

          // 读取sparse参数
          for (size_t param_id = 0; param_id < params_.size(); ++param_id) {
            auto& param = params_[param_id];
            auto& current_csr = host_sparse_buffer_[param_id];
            for (int k = 0; k < param.slot_num; k++) {
              int nnz; // 读取一个int到nnz，就是得到nnz的大小，non-zero feature number
              CK_THROW_(checker_->read(reinterpret_cast<char*>(&nnz), sizeof(int)),
                        "failure in reading nnz");
              if (nnz > (int)buffer_length_ || nnz < 0) {
                ERROR_MESSAGE_(
                    "nnz > buffer_length_ | nnz < 0 nnz:" + std::to_string(nnz) +
                    ". Please check if i64_input_key in config is compatible with dataset");
              }
              current_csr.new_row();  // 换行
              size_t num_value = current_csr.get_num_values();
              // 读取nnz个数据
              CK_THROW_(checker_->read(reinterpret_cast<char*>(
                                           current_csr.get_value_tensor().get_ptr() + num_value),
                                       sizeof(T) * nnz),
                        "failure in reading feature_ids_");
              current_csr.update_value_size(nnz);
            }
          }
        } catch (const internal_runtime_error& rt_err) {  // 回退
          batch_idx--;  // restart i-th sample
          for (auto& each_csr : host_sparse_buffer_) {
            each_csr.roll_back();
          }
          Error_t err = rt_err.get_error();
          if (err == Error_t::DataCheckError) {
            ERROR_MESSAGE_("Error_t::DataCheckError");
          } else {            // Error_t::BrokenFile, Error_t::UnspecificEror, ...
            read_new_file();  // can throw Error_t::EOF
          }
        }

        current_record_index_++;

        // start a new file when finish one file read
        if (current_record_index_ >= data_set_header_.number_of_records) {
          read_new_file();  // can throw Error_t::EOF
        }
      } catch (const internal_runtime_error& rt_err) {
        Error_t err = rt_err.get_error();
        if (err == Error_t::EndOfFile) {
          current_batch_size = batch_idx + 1;
        } else {
          throw;
        }
      }
    }

    for (auto& each_csr : host_sparse_buffer_) {
      each_csr.new_row();
    }
    // do h2d
    // wait buffer and schedule

    // 目前数据在 host_sparse_buffer_（CPU）之上，
    // 需要拷贝到 GPU（目标是 ThreadBuffer 的 device_sparse_buffers 成员变量），
    // 使用 cudaMemcpyHostToDevice

    // 而且，host_sparse_buffer_ 是 CSR<T> 格式，
    // ThreadBuffer 的 device_sparse_buffers 成员变量是SparseTensor<T>格式，需要转换

    if (!wait_until_h2d_ready()) return;
    buffer_->current_batch_size = current_batch_size;
    {
      CudaCPUDeviceContext context(gpu_resource_->get_device_id());
      // 目标是 ThreadBuffer 的 device_sparse_buffers 成员变量
      auto dst_dense_tensor = Tensor2<float>::stretch_from(buffer_->device_dense_buffers);
      CK_CUDA_THROW_(cudaMemcpyAsync(dst_dense_tensor.get_ptr(), host_dense_buffer_.get_ptr(),
                                     host_dense_buffer_.get_size_in_bytes(), cudaMemcpyHostToDevice,
                                     gpu_resource_->get_memcpy_stream()));

      for (size_t param_id = 0; param_id < params_.size(); ++param_id) {  // 遍历嵌入层
        auto dst_sparse_tensor =
            SparseTensor<T>::stretch_from(buffer_->device_sparse_buffers[param_id]);
        if (buffer_->is_fixed_length[param_id] &&
            last_batch_nnz_[param_id] == host_sparse_buffer_[param_id].get_num_values()) {
          // 拷贝到GPU，同时也进行了转换，提取了CSR的成员变量，拷贝到了SparseTensor的对应地址
          CK_CUDA_THROW_(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                         host_sparse_buffer_[param_id].get_value_tensor().get_ptr(),
                                         host_sparse_buffer_[param_id].get_num_values() * sizeof(T),
                                         cudaMemcpyHostToDevice,
                                         gpu_resource_->get_memcpy_stream()));
        } else {
          // 拷贝到GPU
          sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, host_sparse_buffer_[param_id],
                                                 gpu_resource_->get_memcpy_stream());
          last_batch_nnz_[param_id] = host_sparse_buffer_[param_id].get_num_values();
        }
      }
      // 进行同步
      CK_CUDA_THROW_(cudaStreamSynchronize(gpu_resource_->get_memcpy_stream()));
    }
    assert(buffer_->state.load() == BufferState::Writing);
    buffer_->state.store(BufferState::ReadyForRead);
  }
};

}  // namespace HugeCTR
