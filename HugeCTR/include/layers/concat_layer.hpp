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

#include "HugeCTR/include/layer.hpp"

#include <vector>

namespace HugeCTR {

/**
 * Layer which merges the multiple 2D input tensors to a single 2D output tensor.
 * The input tensors and the resulting output tensor must have the same dimensionallity.
 * Only the innermost dimension is expanded by concatenating those of the input tensors.
 * e.g., 3X(batch_size, n_slots * vector_length) to (batch_size, 3 * n_slots * vector_length),
 * e.g., (batch_size, a * vector_length) + (batch_size, b * vector_length)
 *       to (batch_size, (a + b) * vector_length)
 */
template <typename T>
class ConcatLayer : public Layer {
  /*
   * stores the weight tensors of this layer.
   */
  Tensors<T> weights_;
  /*
   * stores the weight gradient tensors of this layer.
   */
  Tensors<T> wgrad_;
  /*
   * stores the references to the input tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<T>>> in_tensors_;
  /*
   * stores the references to the output tensors of this layer.
   */
  std::vector<std::shared_ptr<Tensor<T>>> out_tensors_;

 public:
  /**
   * Ctor of ConcatLayer.
   * @param in_tensors the vector of the input tensors
   * @param out_tensor the resulting output tensor
   * @param blobs_buff GeneralBuffer used to create the output tensor
   * @param device_id the id of GPU where this layer belongs
   */
  ConcatLayer(Tensors<T> in_tensors, std::shared_ptr<Tensor<T>>& out_tensor,
              const std::shared_ptr<GeneralBuffer<T>>& blobs_buff, int device_id);
  ~ConcatLayer() override{};

  /**
   * Concat's foward pass to gather data to the output tensor
   * @param stream CUDA stream where the foward propagation is executed
   */
  void fprop(cudaStream_t stream) override;
  /**
   * Concat's backward pass to scatter data to the input tensors
   * @param stream CUDA stream where the foward propagation is executed
   */
  void bprop(cudaStream_t stream) override;

  struct InParam {
    T* in;
    const int in_w;
  };

 private:
  void prop_common(bool forward, cudaStream_t stream);
  std::vector<InParam> set_in_params(int n);
  template <typename... Args>
  void kernel_launch(bool forward, cudaStream_t stream, Args&... args);

  int n_sms_;
};

}  // namespace HugeCTR
