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
#include <cuda_runtime_api.h>
#include <memory>
#include "tensor2.hpp"

namespace HugeCTR {
// HostAllocator 作用是在host之上管理内存。  后面几个实现都是调用了CUDA函数来进行内存分配，比如 cudaHostAlloc，有兴趣读者可以深入学习
class HostAllocator {
 public:
  void *allocate(size_t size) const { return malloc(size); }
  void deallocate(void *ptr) const { free(ptr); }
};

//调用CUDA方法在主机上分配内存
class CudaHostAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFreeHost(ptr)); }
};

//cudaMallocManaged 分配旨在供主机或设备代码使用的内存，算是一种统一分配内存的方法。
class CudaManagedAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaMallocManaged(&ptr, size));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFree(ptr)); }
};

//该类是在设备上分配内存。
class CudaAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA(cudaMalloc(&ptr, size));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA(cudaFree(ptr)); }
};

template <typename T>
class BufferBlock2 {
 public:
  virtual ~BufferBlock2() {}
  virtual void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) = 0;
  virtual Tensor2<T> &as_tensor() = 0;
};
/*
 4.2.2 GeneralBuffer2
分析完如何分配内存，我们接下来看看如何封装内存，具体通过 GeneralBuffer2 完成的。GeneralBuffer2 可以认为是一个对大段内存的统一封装，具体在其上可以有若干Tensor。

4.2.2.1 定义
   这里都忽略了成员函数，内部类也忽略了成员函数。

       allocator ：具体内存分配器，也区分在GPU分配还是CPU分配。
       ptr_ ：指向分配的内存；
       total_size_in_bytes_ ：内存大小；
       reserved_buffers_ ：前期预留buffer，后续会统一分配;
具体内部类为：
        BufferInternal 是接口。
        TensorBufferImpl 是 Tensor2 对应的buffer实现。
        BufferBlockImpl 则是在构建网络时候会用到。
*/
template <typename Allocator>
class GeneralBuffer2 : public std::enable_shared_from_this<GeneralBuffer2<Allocator>> {
  class BufferInternal {
   public:
    virtual ~BufferInternal() {}
    virtual size_t get_size_in_bytes() const = 0;
    virtual void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) = 0;
  };

  class TensorBufferImpl : public TensorBuffer2, public BufferInternal {
    size_t size_in_bytes_;
    std::shared_ptr<GeneralBuffer2> buffer_;
    size_t offset_;

   public:
    TensorBufferImpl(size_t size_in_bytes) : size_in_bytes_(size_in_bytes) {}
    bool allocated() const override { return buffer_ && buffer_->allocated(); }
    void *get_ptr() override { return forward_void_pointer(buffer_->ptr_, offset_); }

    size_t get_size_in_bytes() const { return size_in_bytes_; }
    //就是指向了一个 GeneralBuffer2，然后设定了自己的offset和大小。
    void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) {
      buffer_ = buffer;
      offset_ = offset;
    }
  };

  template <typename T>
  class BufferBlockImpl : public BufferBlock2<T>, public BufferInternal {
    size_t total_num_elements_;
    std::shared_ptr<TensorBufferImpl> buffer_impl_;
    Tensor2<T> tensor_;
    bool finalized_;
    std::vector<std::shared_ptr<BufferInternal>> reserved_buffers_;

   public:
    BufferBlockImpl() : total_num_elements_(0), finalized_(false) {}

    //BufferBlockImpl 多了一个reserve方法，用来预留内存空间，在此空间之上生成内部tensor。
    void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) override {
      if (finalized_) {
        throw std::runtime_error(ErrorBase + "Buffer block is finalized.");
      }
      size_t num_elements = get_num_elements_from_dimensions(dimensions);
      size_t size_in_bytes = num_elements * TensorScalarSizeFunc<T>::get_element_size();

      std::shared_ptr<TensorBufferImpl> buffer_impl =
          std::make_shared<TensorBufferImpl>(size_in_bytes);
      reserved_buffers_.push_back(buffer_impl);

      *tensor = Tensor2<T>(dimensions, buffer_impl);

      total_num_elements_ += num_elements;
    }

    Tensor2<T> &as_tensor() override {
      if (!finalized_) {
        buffer_impl_ = std::make_shared<TensorBufferImpl>(
            total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size());
        tensor_ = Tensor2<T>({total_num_elements_}, buffer_impl_);
        finalized_ = true;
      }
      return tensor_;
    };

    size_t get_size_in_bytes() const {
      return total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size();
    }

    //initialize 会对内部进行配置
    void initialize(const std::shared_ptr<GeneralBuffer2> &buffer, size_t offset) {
      size_t local_offset = 0;
      for (const std::shared_ptr<BufferInternal> &buffer_impl : reserved_buffers_) {
        buffer_impl->initialize(buffer, offset + local_offset);
        local_offset += buffer_impl->get_size_in_bytes();
      }
      reserved_buffers_.clear();

      if (!finalized_) {
        buffer_impl_ = std::make_shared<TensorBufferImpl>(
            total_num_elements_ * TensorScalarSizeFunc<T>::get_element_size());
        tensor_ = Tensor2<T>({total_num_elements_}, buffer_impl_);
        finalized_ = true;
      }
      buffer_impl_->initialize(buffer, offset);
    }
  };

  Allocator allocator_;
  void *ptr_;
  size_t total_size_in_bytes_;
  std::vector<std::shared_ptr<BufferInternal>> reserved_buffers_;

  GeneralBuffer2() : ptr_(nullptr), total_size_in_bytes_(0) {}

 public:
  static std::shared_ptr<GeneralBuffer2> create() {
    return std::shared_ptr<GeneralBuffer2>(new GeneralBuffer2);
  }

  GeneralBuffer2(const GeneralBuffer2 &) = delete;
  GeneralBuffer2 &operator=(const GeneralBuffer2 &) = delete;

  ~GeneralBuffer2() {
    if (allocated()) {
      allocator_.deallocate(ptr_);
    }
  }

  //allocate 会遍历注册的 BufferInternal，累积其总大小，最后调用 allocator_ 进行分配内存
  void allocate() {
    if (ptr_ != nullptr) {
      throw std::runtime_error(ErrorBase + "Memory has already been allocated.");
    }

    size_t offset = 0;
    for (const std::shared_ptr<BufferInternal> &buffer : reserved_buffers_) {
      buffer->initialize(this->shared_from_this(), offset);
      size_t size_in_bytes = buffer->get_size_in_bytes();
      if (size_in_bytes % 32 != 0) {
        size_in_bytes += (32 - size_in_bytes % 32);
      }
      offset += size_in_bytes;
    }
    reserved_buffers_.clear();
    total_size_in_bytes_ = offset;

    if (total_size_in_bytes_ != 0) {
      ptr_ = allocator_.allocate(total_size_in_bytes_);
    }
  }

  //create_block 会针对BufferBlock2进行创建
  template <typename T>
  std::shared_ptr<BufferBlock2<T>> create_block() {
    if (allocated()) {
      throw std::runtime_error(ErrorBase + "General buffer is finalized.");
    }
    std::shared_ptr<BufferBlockImpl<T>> block_impl = std::make_shared<BufferBlockImpl<T>>();
    reserved_buffers_.push_back(block_impl);
    return block_impl;
  }

  //reserve 方法会把某一个张量对应的内存需求用 TensorBufferImpl 的形式记录在reserved_buffers_之中，然后生成这个张量，而且就是用TensorBufferImpl 生成。
  template <typename T>
  void reserve(const std::vector<size_t> &dimensions, Tensor2<T> *tensor) {
    if (allocated()) {
      throw std::runtime_error(ErrorBase + "General buffer is finalized.");
    }

    size_t size_in_bytes =
        get_num_elements_from_dimensions(dimensions) * TensorScalarSizeFunc<T>::get_element_size();

    std::shared_ptr<TensorBufferImpl> buffer_impl =
        std::make_shared<TensorBufferImpl>(size_in_bytes);
    reserved_buffers_.push_back(buffer_impl);

    *tensor = Tensor2<T>(dimensions, buffer_impl);
  }

  bool allocated() const { return total_size_in_bytes_ != 0 && ptr_ != nullptr; }

};  // namespace HugeCTR

}  // namespace HugeCTR