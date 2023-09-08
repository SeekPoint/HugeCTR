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
#include <memory>
#include <numeric>
#include <vector>
namespace HugeCTR {

enum class TensorScalarType { None, Void, Float32, Float16, Int64, UInt64, Int32, UInt32, Size_t };

namespace {

inline size_t get_num_elements_from_dimensions(const std::vector<size_t> &dimensions) {
  size_t elements = 1;
  for (size_t dim : dimensions) {
    elements *= dim;
  }
  return elements;
}

inline void *forward_void_pointer(void *ptr, size_t offset) {
  return reinterpret_cast<unsigned char *>(ptr) + offset;
}

template <typename T>
struct TensorScalarSizeFunc {
  static size_t get_element_size() { return sizeof(T); }
};

template <>
struct TensorScalarSizeFunc<void> {
  static size_t get_element_size() { return 1ul; }
};

template <typename T>
struct TensorScalarTypeFunc {};

template <>
struct TensorScalarTypeFunc<float> {
  static TensorScalarType get_type() { return TensorScalarType::Float32; }
};

template <>
struct TensorScalarTypeFunc<__half> {
  static TensorScalarType get_type() { return TensorScalarType::Float16; }
};

template <>
struct TensorScalarTypeFunc<size_t> {
  static TensorScalarType get_type() { return TensorScalarType::Size_t; }
};

template <>
struct TensorScalarTypeFunc<long long> {
  static TensorScalarType get_type() { return TensorScalarType::Int64; }
};

template <>
struct TensorScalarTypeFunc<unsigned int> {
  static TensorScalarType get_type() { return TensorScalarType::UInt32; }
};

}  // namespace
//TensorBuffer2 是张量底层的数据，也许联系到 PyTorch 的 data 或者 storage 可以更好的理解
class TensorBuffer2 {
 public:
  virtual ~TensorBuffer2() {}
  virtual bool allocated() const = 0;
  virtual void *get_ptr() = 0;
};

/*4.1.4 TensorBag2
PyTorch 之中也有一些Bag后缀名字的类，比如 nn.Embedding和nn.EmbeddingBag。
 当构建袋子模型时，做一个Embedding跟随Sum或是Mean常见的。
 对于可变长度序列，nn.EmbeddingBag 来提供了更加高效和更快速的处理方式，特别是对于可变长度序列。

在 HugeCTR，TensorBag2 可以认为是把 Tensor 放在袋子里统一处理的类
 */
class TensorBag2 {
  template <typename T>
  friend class Tensor2;

  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> buffer_;
  TensorScalarType scalar_type_;

  TensorBag2(const std::vector<size_t> dimensions, const std::shared_ptr<TensorBuffer2> &buffer,
             TensorScalarType scalar_type)
      : dimensions_(dimensions), buffer_(buffer), scalar_type_(scalar_type) {}

 public:
  TensorBag2() : scalar_type_(TensorScalarType::None) {}

  const std::vector<size_t> &get_dimensions() const { return dimensions_; }

  void *get_ptr() { return buffer_->get_ptr(); }
};
//以下是两个向量类，用来方便用户使用
using TensorBags2 = std::vector<TensorBag2>;

template <typename T>
class Tensor2 {
  std::vector<size_t> dimensions_;
  size_t num_elements_;
  std::shared_ptr<TensorBuffer2> buffer_;

 public:
  static Tensor2 stretch_from(const TensorBag2 &bag) {
    if (bag.scalar_type_ != TensorScalarTypeFunc<T>::get_type()) {
      CK_THROW_(Error_t::WrongInput, "Inconsistent tensor type");
    }

    return Tensor2(bag.dimensions_, bag.buffer_);
  }

  Tensor2() : num_elements_(0) {}

  Tensor2(Tensor2 const &other) = default;
  Tensor2 &operator=(Tensor2 const &other) = default;

  Tensor2(Tensor2 &&other) = default;
  Tensor2 &operator=(Tensor2 &&other) = default;

  Tensor2(const std::vector<size_t> &dimensions, const std::shared_ptr<TensorBuffer2> &buffer)
      : dimensions_(dimensions),
        num_elements_(get_num_elements_from_dimensions(dimensions)),
        buffer_(buffer) {}

  TensorBag2 shrink() const {
    return TensorBag2(dimensions_, buffer_, TensorScalarTypeFunc<T>::get_type());
  }

  bool allocated() const { return buffer_ && buffer_->allocated(); }

  const std::vector<size_t> &get_dimensions() const { return dimensions_; }

  size_t get_num_elements() const { return num_elements_; }

  size_t get_size_in_bytes() const {
    return num_elements_ * TensorScalarSizeFunc<T>::get_element_size();
  }

  void set_buffer(const std::shared_ptr<TensorBuffer2> &buffer) { buffer_ = buffer; }

  std::shared_ptr<TensorBuffer2> get_buffer() const { return buffer_; }

  const T *get_ptr() const { return reinterpret_cast<const T *>(buffer_->get_ptr()); }

  T *get_ptr() { return reinterpret_cast<T *>(buffer_->get_ptr()); }

  void reset_shape(const std::vector<size_t> &new_dimensions) {
    try {
      size_t new_num_elements = get_num_elements_from_dimensions(new_dimensions);
      if (new_num_elements > num_elements_) {
        CK_THROW_(Error_t::WrongInput, "new dimensions out of memory");
      }
      dimensions_ = new_dimensions;
      num_elements_ = new_num_elements;
    } catch (const std::runtime_error &rt_err) {
      std::cerr << rt_err.what() << std::endl;
    }
  }
};

template <typename T>
using Tensors2 = std::vector<Tensor2<T>>; //Tensors2 就是 Tensor2 的一个vector。

//关于 Tensor 和 Bag 的联系，可以参见下面的函数。
template <typename T>
Tensors2<T> bags_to_tensors(const std::vector<TensorBag2> &bags) {
  Tensors2<T> tensors;
  for (const auto &bag : bags) {
    tensors.push_back(Tensor2<T>::stretch_from(bag));
  }
  return tensors;
}

template <typename T>
std::vector<TensorBag2> tensors_to_bags(const Tensors2<T> &tensors) {
  std::vector<TensorBag2> bags;
  for (const auto &tensor : tensors) {
    bags.push_back(tensor.shrink());
  }
  return bags;
}

//类似 TensorBag 的功能
class SparseTensorBag {
  template <typename T>
  friend class SparseTensor;

  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> value_buffer_;
  std::shared_ptr<TensorBuffer2> rowoffset_buffer_;
  std::shared_ptr<size_t> nnz_;
  size_t rowoffset_count_;
  TensorScalarType scalar_type_;

  SparseTensorBag(const std::vector<size_t> &dimensions,
                  const std::shared_ptr<TensorBuffer2> &value_buffer,
                  const std::shared_ptr<TensorBuffer2> &rowoffset_buffer,
                  const std::shared_ptr<size_t> &nnz, const size_t rowoffset_count,
                  TensorScalarType scalar_type)
      : dimensions_(dimensions),
        value_buffer_(value_buffer),
        rowoffset_buffer_(rowoffset_buffer),
        nnz_(nnz),
        rowoffset_count_(rowoffset_count),
        scalar_type_(scalar_type) {}

 public:
  SparseTensorBag() : scalar_type_(TensorScalarType::None) {}
  const std::vector<size_t> &get_dimensions() const { return dimensions_; }
};

/*4.1.5 SparseTensor
SparseTensor 是 Sparse 类型的张量，这是3.2 版本加入的，
 目的是为了统一处理CSR格式，或者说是统一处理稀疏矩阵，
 可以有效存储和处理大多数元素为零的张量。后续在读取数据到GPU时候会有分析。
 我们对比一下 CSR 格式，就可以看出来其内部机制就对应了CSR 的 rowoffset 和 value。其具体定义如下：
 示意图如下： 002-004.jpg
我们从中找出一个例子看看。因为只是用来存储slot里的sparse key，所以没有列号，因为一个slot里的sparse key可以直接顺序存储。

   * For example data:
   *   4,5,1,2
       *   3,5,1
       *   3,2
       * Will be convert to the form of:
   * row offset: 0,4,7,9
       * value: 4,5,1,2,3,5,1,3,2
       对应下图： 002-005.jpg

PyTorch
PyTorch 有 sparse_coo_tensor 可以实现类似的功能。PyTorch 支持不同layout的张量，
 大家可以从 torch/csrc/utils/tensor_layouts.cpp 找到，
 比如 at::Layout::Strided，at::Layout::Sparse，at::Layout::SparseCsr，at::Layout::Mkldnn 等等，
 这些对应了不同的内存布局模式。

使用稀疏张量时候，提供一对 dense tensors：一个value张量，一个二维indice张量，也有其他辅助参数。

    >>> i = [[1, 1]]
    >>> v =  [3, 4]
    >>> s=torch.sparse_coo_tensor(i, v, (3,))
    >>> s
                                tensor(indices=tensor([[1, 1]]),
                                       values=tensor(  [3, 4]),
                                       size=(3,), nnz=2, layout=torch.sparse_coo)
                                    TensorFlow
    TensorFlow 也有 SparseTensor 类型来表示多维稀疏数据。一个 SparseTensor 使用三个稠密张量来表示：

                            indices 表示稀疏张量的非零元素坐标。
                            values 则对应每个非零元素的值。
                            shape 表示本稀疏张量转换为稠密形式后的形状。
                            比如下面代码：

                            indices = tf.constant([[0, 0], [1, 1], [2,2]], dtype=tf.int64)
                            values = tf.constant([1, 2, 3], dtype=tf.float32)
                            shape = tf.constant([3, 3], dtype=tf.int64)
                            sparse = tf.SparseTensor(indices=indices,
                                                     values=values,
                                                     dense_shape=shape)
                            dense = tf.sparse_tensor_to_dense(sparse, default_value=0)
                            with tf.Session() as session:
                                sparse, dense = session.run([sparse, dense])
                                print('Sparse is :\n', sparse)
                                print('Dense is :\n', dense)
         打印出来如下：
                Sparse is :
    SparseTensorValue(indices=array([[0, 0],
                                        [1, 1],
                                        [2, 2]]), values=array([1., 2., 3.], dtype=float32), dense_shape=array([3, 3]))
        Dense is :
               [[1. 0. 0.]
              [0. 2. 0.]
              [0. 0. 3.]]
            */
template <typename T>
class SparseTensor {
  std::vector<size_t> dimensions_;
  std::shared_ptr<TensorBuffer2> value_buffer_;
  std::shared_ptr<TensorBuffer2> rowoffset_buffer_;
  std::shared_ptr<size_t> nnz_;  // maybe size_t for FixedLengthSparseTensor
  size_t rowoffset_count_;

 public:
  SparseTensor() {}
  SparseTensor(const std::vector<size_t> &dimensions,
               const std::shared_ptr<TensorBuffer2> &value_buffer,
               const std::shared_ptr<TensorBuffer2> &rowoffset_buffer,
               const std::shared_ptr<size_t> &nnz, const size_t rowoffset_count)
      : dimensions_(dimensions),
        value_buffer_(value_buffer),
        rowoffset_buffer_(rowoffset_buffer),
        nnz_(nnz),
        rowoffset_count_(rowoffset_count) {}

  SparseTensor(const Tensor2<T> &value_tensor, const Tensor2<T> &rowoffset_tensor,
               const std::shared_ptr<size_t> nnz)
      : dimensions_(value_tensor.get_dimensions()),
        value_buffer_(value_tensor.get_buffer()),
        rowoffset_buffer_(rowoffset_tensor.get_buffer()),
        nnz_(nnz),
        rowoffset_count_(rowoffset_tensor.get_num_elements()) {}

  static SparseTensor stretch_from(const SparseTensorBag &bag) {
    if (bag.scalar_type_ != TensorScalarTypeFunc<T>::get_type()) {
      CK_THROW_(Error_t::WrongInput, "Inconsistent sparse tensor type");
    }
    return SparseTensor(bag.dimensions_, bag.value_buffer_, bag.rowoffset_buffer_, bag.nnz_,
                        bag.rowoffset_count_);
  }

  SparseTensorBag shrink() const {
    return SparseTensorBag(dimensions_, value_buffer_, rowoffset_buffer_, nnz_, rowoffset_count_,
                           TensorScalarTypeFunc<T>::get_type());
  }

  T *get_value_ptr() { return reinterpret_cast<T *>(value_buffer_->get_ptr()); }

  const T *get_value_ptr() const { return reinterpret_cast<const T *>(value_buffer_->get_ptr()); }

  Tensor2<T> get_value_tensor() const { return Tensor2<T>(dimensions_, value_buffer_); }

  T *get_rowoffset_ptr() { return reinterpret_cast<T *>(rowoffset_buffer_->get_ptr()); }

  const T *get_rowoffset_ptr() const {
    return reinterpret_cast<const T *>(rowoffset_buffer_->get_ptr());
  }

  Tensor2<T> get_rowoffset_tensor() const {
    return Tensor2<T>({rowoffset_count_}, rowoffset_buffer_);
  }

  const std::vector<size_t> &get_dimensions() const { return dimensions_; }

  size_t max_nnz() const { return get_num_elements_from_dimensions(dimensions_); }

  size_t nnz() const { return *nnz_; }

  std::shared_ptr<size_t> get_nnz_ptr() { return nnz_; }

  size_t rowoffset_count() const { return rowoffset_count_; }
};
//以下是向量类，用来方便用户使用。
template <typename T>
using SparseTensors = std::vector<SparseTensor<T>>;

template <typename T>
class CSR;
namespace sparse_tensor_helper {
namespace cuda {
template <typename T>
void copy_async(SparseTensor<T> &dst, const SparseTensor<T> &src, cudaMemcpyKind kind,
                cudaStream_t stream) {
  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_value_ptr(), src.get_value_ptr(), src.nnz() * sizeof(T),
                                 kind, stream));

  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_rowoffset_ptr(), src.get_rowoffset_ptr(),
                                 src.rowoffset_count() * sizeof(T), kind, stream));

  *dst.get_nnz_ptr() = src.nnz();
}

template <typename T>
void copy_async(SparseTensor<T> &dst, const CSR<T> &src, cudaStream_t stream) {
  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_value_ptr(), src.get_value_tensor().get_ptr(),
                                 src.get_num_values() * sizeof(T), cudaMemcpyHostToDevice, stream));

  CK_CUDA_THROW_(cudaMemcpyAsync(dst.get_rowoffset_ptr(), src.get_row_offset_tensor().get_ptr(),
                                 src.get_row_offset_tensor().get_size_in_bytes(),
                                 cudaMemcpyHostToDevice, stream));

  *dst.get_nnz_ptr() = src.get_num_values();
}
}  // namespace cuda
namespace cpu {
template <typename T>
void copy(SparseTensor<T> &dst, const SparseTensor<T> &src) {
  memcpy(dst.get_value_ptr(), src.get_value_ptr(), src.nnz() * sizeof(T));
  memcpy(dst.get_rowoffset_ptr(), src.get_rowoffset_ptr(), src.rowoffset_count() * sizeof(T));

  *dst.get_nnz_ptr() = src.nnz();
}
}  // namespace cpu
}  // namespace sparse_tensor_helper
}  // namespace HugeCTR
