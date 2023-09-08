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

#include <unistd.h>

#include <atomic>
#include <common.hpp>
#include <memory>
#include <mutex>
#include <queue>
#include <resource_manager.hpp>
#include <thread>
#include <utils.hpp>

#include "data_readers/data_reader_common.hpp"
#ifdef ENABLE_MPI
#include <mpi.h>
#endif
namespace HugeCTR {

template <typename TypeComp>
void split(Tensor2<float> &label_tensor, Tensor2<TypeComp> &dense_tensor,
           const Tensor2<float> &label_dense_buffer, const int label_dense_dim,
           cudaStream_t stream);

/*
 0x05 读取到embedding
我们接下来看看 DataCollector，就是流水线的第二级，就是这里的黄色框 "Copy to GPU"。其实其内部文字修改为：Copy To Embedding 更合适。
 003-009.png
此图显示了“读取文件”、“复制到 GPU”和“训练”阶段如何重叠三个批次以提高 GPU 资源利用率。
5.1 DataCollector
我们首先看看DataCollector的定义，这里省略了成员函数，主要成员变量是。
    std::shared_ptr broadcast_buffer_ : CPU 数据拷贝到 GPU 之上，GPU 上就在这里。
    std::shared_ptr output_buffer_ ：这个就是 DataReaderOutput，就是 Reader 的成员变量，复制到这里是为了 collector 操作方便。
    BackgroundDataCollectorThread background_collector_ ：线程主体，主要包括 ThreadBuffer 和 BroadcastBuffer，会把数据从 ThreadBuffer 拷贝到 BroadcastBuffer 之上。
    std::thread background_collector_thread_ ：工作线程。
目前具体如下，Collector 之中的 broadcast_buffer_ 和 output_buffer_ 都指向了GPU，但GPU之中尚且没有数据：
 003-010.jpg
 */
/**
 * @brief A helper class of data reader.
 *
 * This class implement asynchronized data collecting from heap
 * to output of data reader, thus data collection and training
 * can work in a pipeline.
 5.2 ThreadBuffer 2 BroadBuffer
5.2.1 工作线程
BackgroundDataCollectorThread 的作用是把数据从 DataReader 的thread_buffers_拷贝到 broadcast_buffer_。
 */
template <typename T>
void broadcast(const std::shared_ptr<ThreadBuffer> &thread_buffer,
               std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
               std::vector<size_t> &last_batch_nnz_,
               const std::shared_ptr<ResourceManager> &resource_manager);

/**
 * @brief A helper class of data reader.
 *
 * This class implement asynchronized data collecting from heap
 * to output of data reader, thus data collection and training
 * can work in a pipeline.
 */
template <typename T>
class DataCollector {
  class BackgroundDataCollectorThread {
    std::vector<std::shared_ptr<ThreadBuffer>> thread_buffers_;
    std::shared_ptr<BroadcastBuffer> broadcast_buffer_;

    std::atomic<bool> loop_flag_;
    int counter_;
    std::vector<size_t> last_batch_nnz_;  // local_gpu_count * embedding number
    std::vector<char> worker_status_;
    int eof_worker_num_;

    std::shared_ptr<ResourceManager> resource_manager_;

   public:
    BackgroundDataCollectorThread(const std::vector<std::shared_ptr<ThreadBuffer>> &thread_buffers,
                                  const std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
                                  const std::shared_ptr<ResourceManager> &resource_manager)
        : thread_buffers_(thread_buffers),
          broadcast_buffer_(broadcast_buffer),
          loop_flag_{true},
          counter_{0},
          last_batch_nnz_(
              broadcast_buffer->is_fixed_length.size() * resource_manager->get_local_gpu_count(),
              0),
          worker_status_(thread_buffers.size(), 0),
          eof_worker_num_(0),
          resource_manager_(resource_manager) {}

    void start() {
      while (loop_flag_.load()) {
        // threadbuffer是源数据，broadcast buffer是目标数据
        auto &current_src_buffer = thread_buffers_[counter_];
        // auto &next_src_buffer = thread_buffers_[(counter_ + 1) % thread_buffers_.size()];
        auto &dst_buffer = broadcast_buffer_;
        auto src_expected = BufferState::ReadyForRead; // 期望源数据是这个状态
        auto dst_expected = BufferState::ReadyForWrite; // 期望目标数据是这个状态

        if (worker_status_[counter_]) {
          counter_ = (counter_ + 1) % thread_buffers_.size();
          continue;
        }

        if ((current_src_buffer->state.load() == BufferState::Reading ||
             current_src_buffer->state.compare_exchange_weak(src_expected, BufferState::Reading)) &&
            (dst_buffer->state.load() == BufferState::Writing ||
             dst_buffer->state.compare_exchange_weak(dst_expected, BufferState::Writing))) {
          assert(current_src_buffer->state.load() == BufferState::Reading);
          assert(dst_buffer->state.load() == BufferState::Writing);

          // 如果源数据是可读或者正在读，并且，目标数据是可写或者正在写，则可以操作
          if (current_src_buffer->current_batch_size == 0) {
            worker_status_[counter_] = 1;
            eof_worker_num_ += 1;
            current_src_buffer->state.store(BufferState::FileEOF);
          }
          if (static_cast<size_t>(eof_worker_num_) != thread_buffers_.size() &&
              current_src_buffer->current_batch_size == 0) {
            counter_ = (counter_ + 1) % thread_buffers_.size();
            dst_buffer->state.store(BufferState::ReadyForWrite);  // 设定目标数据的状态
            continue;
          }
          dst_buffer->current_batch_size = current_src_buffer->current_batch_size;
          if (current_src_buffer->current_batch_size != 0) {
            // 进行广播操作
            broadcast<T>(current_src_buffer, dst_buffer, last_batch_nnz_, resource_manager_);

            current_src_buffer->state.store(BufferState::ReadyForWrite); // 设定目标数据的状态
            counter_ = (counter_ + 1) % thread_buffers_.size();
          } else {
            memset(worker_status_.data(), 0, sizeof(char) * worker_status_.size());
            eof_worker_num_ = 0;
            counter_ = 0;
          }

          // 会通知源数据可以继续读取了
          dst_buffer->state.store(BufferState::ReadyForRead);
        } else {
          usleep(2);  // 否则等待一会
        }
      }
    }

    void stop() { loop_flag_.store(false); }
  };

  std::shared_ptr<BroadcastBuffer> broadcast_buffer_;
  std::shared_ptr<DataReaderOutput> output_buffer_;

  BackgroundDataCollectorThread background_collector_;
  std::thread background_collector_thread_;

  std::atomic<bool> loop_flag_;
  std::vector<size_t> last_batch_nnz_;

  std::shared_ptr<ResourceManager> resource_manager_;

 public:
  void stop() { background_collector_.stop(); }
  DataCollector(const std::vector<std::shared_ptr<ThreadBuffer>> &thread_buffers,
                const std::shared_ptr<BroadcastBuffer> &broadcast_buffer,
                std::shared_ptr<DataReaderOutput> &output,
                const std::shared_ptr<ResourceManager> &resource_manager)
      : broadcast_buffer_(broadcast_buffer),
        output_buffer_(output),
        background_collector_(thread_buffers, broadcast_buffer, resource_manager),
        loop_flag_{true},
        last_batch_nnz_(
            broadcast_buffer->is_fixed_length.size() * resource_manager->get_local_gpu_count(), 0),
        resource_manager_(resource_manager) {
    background_collector_thread_ = std::thread([this]() { background_collector_.start(); });
  }

  ~DataCollector() {
    background_collector_.stop();
    background_collector_thread_.join();
  }
//我们看看 read_a_batch_to_device。这里 read_a_batch_to_device_delay_release 和 read_a_batch_to_device 是沿用旧版本命名，已经和目前状况不符合。
//
//具体逻辑是：看看 broadcast_buffer_ 的状态是不是可以读取 ReadyForRead，如果不可以，就等一会。
//如果可以，就继续，即遍历GPU，逐个从broadcast拷贝到output（也是设备之间的拷贝），也对 label 和 dense 进行split。
  long long read_a_batch_to_device() {
    // MESSAGE_("data collector waiting read_a_batch_to_device");
    BufferState expected = BufferState::ReadyForRead;
    while (!broadcast_buffer_->state.compare_exchange_weak(expected, BufferState::Reading)) {
      expected = BufferState::ReadyForRead;
      usleep(2);
    }
    long long current_batch_size = broadcast_buffer_->current_batch_size;
    if (current_batch_size != 0) {
      int local_gpu_count = resource_manager_->get_local_gpu_count();

#pragma omp parallel for num_threads(local_gpu_count)
      for (int i = 0; i < local_gpu_count; ++i) {
        auto local_gpu = resource_manager_->get_local_gpu(i);

        CudaDeviceContext ctx(local_gpu->get_device_id());

        // wait until last iteration finish
        auto label_tensor = Tensor2<float>::stretch_from(output_buffer_->label_tensors[i]);
        auto label_dense_tensor = Tensor2<float>::stretch_from(broadcast_buffer_->dense_tensors[i]);

        // 遍历 sparse 参数
        for (size_t param_id = 0; param_id < output_buffer_->sparse_name_vec.size(); ++param_id) {
          const auto &top_name = output_buffer_->sparse_name_vec[param_id];
          int idx_broadcast = i * broadcast_buffer_->param_num + param_id;
          // broadcast 的是源
          auto src_sparse_tensor =
              SparseTensor<T>::stretch_from(broadcast_buffer_->sparse_buffers[idx_broadcast]);
          if (output_buffer_->sparse_tensors_map.find(top_name) ==
              output_buffer_->sparse_tensors_map.end()) {
            CK_THROW_(Error_t::IllegalCall, "can not find sparse name");
          }
          // output是目标
          auto dst_sparse_tensor =
              SparseTensor<T>::stretch_from(output_buffer_->sparse_tensors_map[top_name][i]);

          // 从broadcast拷贝到output
          if (broadcast_buffer_->is_fixed_length[idx_broadcast] &&
              last_batch_nnz_[idx_broadcast] == src_sparse_tensor.nnz()) {
            CK_CUDA_THROW_(cudaMemcpyAsync(dst_sparse_tensor.get_value_ptr(),
                                           src_sparse_tensor.get_value_ptr(),
                                           src_sparse_tensor.nnz() * sizeof(T),
                                           cudaMemcpyDeviceToDevice, local_gpu->get_stream()));
          } else {
            // 从broadcast拷贝到output
            sparse_tensor_helper::cuda::copy_async(dst_sparse_tensor, src_sparse_tensor,
                                                   cudaMemcpyDeviceToDevice,
                                                   local_gpu->get_stream());
            last_batch_nnz_[idx_broadcast] = src_sparse_tensor.nnz();
          }
        }
        const int label_dense_dim = output_buffer_->label_dense_dim;

        // 拷贝label和dense
        if (output_buffer_->use_mixed_precision) {
          auto dense_tensor = Tensor2<__half>::stretch_from(output_buffer_->dense_tensors[i]);
          // 进行分块
          split(label_tensor, dense_tensor, label_dense_tensor, label_dense_dim,
                local_gpu->get_stream());
        } else {
          auto dense_tensor = Tensor2<float>::stretch_from(output_buffer_->dense_tensors[i]);
          split(label_tensor, dense_tensor, label_dense_tensor, label_dense_dim,
                local_gpu->get_stream());
        }
      }
    } else {
      broadcast_buffer_->state.store(BufferState::ReadyForWrite);
    }
    return current_batch_size;
  }

  //这样后续就可以训练了，后续是通过 finalize_batch 之中进行读取。
  void finalize_batch() {
    for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
      const auto &local_gpu = resource_manager_->get_local_gpu(i);
      CudaDeviceContext context(local_gpu->get_device_id());
      CK_CUDA_THROW_(cudaStreamSynchronize(local_gpu->get_stream()));
    }

    broadcast_buffer_->state.store(BufferState::ReadyForWrite);
  }
};
}  // namespace HugeCTR
