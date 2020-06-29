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

#include "HugeCTR/include/layers/fused_fully_connected_layer.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>
#include "HugeCTR/include/general_buffer.hpp"
#include "cublas_v2.h"
#include "gtest/gtest.h"
#include "utest/test_utils.h"
using namespace std;
using namespace HugeCTR;

static void cpu_mm(__half *c, const __half *a, bool transpose_a, const __half *b, bool transpose_b,
                   int m, int k, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        int ai = transpose_a ? kk * m + i : i * k + kk;
        int bi = transpose_b ? j * k + kk : kk * n + j;
        sum += a[ai] * b[bi];
      }
      c[i * n + j] = sum;
    }
  }
}

static void cpu_add_bias_and_re(__half *top, __half *middle, const __half *bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      __half t = top[i * n + j] + bias[j];
      middle[i * n + j] = t;
      top[i * n + j] = t < 0 ? __float2half(0.0f) : t;
    }
  }
}

static void cpu_reverse_add_bias_and_re(__half *bias_grad, __half *middle, const __half *top, int m,
                                        int n) {
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      if (middle[i * n + j] < 0) {
        middle[i * n + j] = 0.0f;
      } else {
        middle[i * n + j] = top[i * n + j];
      }
    }

  for (int i = 0; i < n; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < m; ++j) sum += middle[j * n + i];
    bias_grad[i] = sum;
  }
}

static float compare_array(const __half *arr1, const __half *arr2, size_t n, float threshold) {
  size_t m = 0;
  for (size_t i = 0; i < n; i++) {
    if (fabs(arr1[i] - arr2[i]) > threshold) {
      m++;
    }
  }
  return 1.0f * m / n;
}

static void fully_connected_layer_test(size_t m, size_t n, size_t k) {
  printf("Testing m=%zu, n=%zu, k=%zu\n", m, n, k);

  GaussianDataSimulator<float> simulator(0.0f, 1.0f, -100.0f, 100.0f);

  GeneralBufferPtr<float> master_weights(new GeneralBuffer<float>());
  GeneralBufferPtr<__half> weights(new GeneralBuffer<__half>());
  GeneralBufferPtr<__half> weights_grad(new GeneralBuffer<__half>());
  GeneralBufferPtr<float> blobs(new GeneralBuffer<float>());
  GeneralBufferPtr<__half> blobs_half(new GeneralBuffer<__half>());

  TensorPtr<__half> bottom_tensor(
      new Tensor<__half>((std::vector<size_t>){m, k}, blobs_half, TensorFormat_t::HW));
  TensorPtr<__half> top_tensor(
      new Tensor<__half>((std::vector<size_t>){m, n}, blobs_half, TensorFormat_t::HW));

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  FusedFullyConnectedLayer fully_connected_layer(master_weights, weights, weights_grad, blobs,
                                                 blobs_half, bottom_tensor, top_tensor,
                                                 TensorFormat_t::HW, cublas_handle, 0);

  // Initialize tensors to 0 and choose cublas algorithms
  master_weights->init(0);
  weights->init(0);
  weights_grad->init(0);
  blobs->init(0);
  blobs_half->init(0);

  fully_connected_layer.optimize();
  // Reset tensors to 0 to ensure all the data are the same as original utest(clear the side effect
  // of optimize)
  weights->reset_sync();
  weights_grad->reset_sync();
  blobs->reset_sync();
  // TODO: result check
  __half *d_kernel = weights->get_ptr_with_offset(0);
  __half *d_bias = weights->get_ptr_with_offset(k * n);
  __half *d_kernel_grad = weights_grad->get_ptr_with_offset(0);
  __half *d_bias_grad = weights_grad->get_ptr_with_offset(k * n);
  __half *d_bottom = blobs_half->get_ptr_with_offset(0);
  __half *d_top = blobs_half->get_ptr_with_offset(m * k);

  std::unique_ptr<__half[]> h_kernel(new __half[k * n]);
  std::unique_ptr<__half[]> h_kernel_grad(new __half[k * n]);
  std::unique_ptr<__half[]> h_bias_grad(new __half[n]);
  std::unique_ptr<__half[]> h_bottom(new __half[m * k]);
  std::unique_ptr<__half[]> h_middle(new __half[m * n]);
  std::unique_ptr<__half[]> h_top(new __half[m * n]);
  std::unique_ptr<__half[]> h_bias(new __half[n]);

  std::unique_ptr<__half[]> d2h_top(new __half[m * n]);
  std::unique_ptr<__half[]> d2h_bottom(new __half[m * k]);
  std::unique_ptr<__half[]> d2h_kernel_grad(new __half[k * n]);
  std::unique_ptr<__half[]> d2h_bias_grad(new __half[n]);

  for (size_t i = 0; i < m * k; ++i) h_bottom[i] = simulator.get_num();
  for (size_t i = 0; i < k * n; ++i) h_kernel[i] = simulator.get_num();
  for (size_t i = 0; i < n; ++i) h_bias[i] = simulator.get_num();

  cudaMemcpy(d_kernel, h_kernel.get(), sizeof(__half) * k * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias.get(), sizeof(__half) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bottom, h_bottom.get(), sizeof(__half) * m * k, cudaMemcpyHostToDevice);

  // cpu fprop
  cpu_mm(h_top.get(), h_bottom.get(), false, h_kernel.get(), false, m, k, n);
  cpu_add_bias_and_re(h_top.get(), h_middle.get(), h_bias.get(), m, n);

  fully_connected_layer.fprop(cudaStreamDefault);

  cudaMemcpy(d2h_top.get(), d_top, sizeof(__half) * m * n, cudaMemcpyDeviceToHost);

  ASSERT_LT(compare_array(h_top.get(), d2h_top.get(), m * n, 1e-5), 0.01f)
      << "fprop cross_check result fail" << endl;

  for (size_t i = 0; i < m * n; ++i) h_top[i] = simulator.get_num();
  cudaMemcpy(d_top, h_top.get(), sizeof(__half) * m * n, cudaMemcpyHostToDevice);

  cpu_reverse_add_bias_and_re(h_bias_grad.get(), h_middle.get(), h_top.get(), m, n);

  cpu_mm(h_kernel_grad.get(), h_bottom.get(), true, h_middle.get(), false, k, m, n);
  cpu_mm(h_bottom.get(), h_middle.get(), false, h_kernel.get(), true, m, n, k);

  fully_connected_layer.bprop(cudaStreamDefault);

  cudaMemcpy(d2h_bottom.get(), d_bottom, sizeof(__half) * m * k, cudaMemcpyDeviceToHost);
  cudaMemcpy(d2h_kernel_grad.get(), d_kernel_grad, sizeof(__half) * k * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(d2h_bias_grad.get(), d_bias_grad, sizeof(__half) * n, cudaMemcpyDeviceToHost);

  ASSERT_LT(compare_array(h_bottom.get(), d2h_bottom.get(), m * k, 1e-3), 0.05f)
      << " bprop cross_check input_grad fail" << endl;
  ASSERT_LT(compare_array(h_kernel_grad.get(), d2h_kernel_grad.get(), k * n, 1e-1), 0.15f)
      << " bprop cross_check weight_grad fail" << endl;
  ASSERT_LT(compare_array(h_bias_grad.get(), d2h_bias_grad.get(), n, 1e-1), 0.15f)
      << " bprop cross_check bias_grad fail" << endl;
}

TEST(layers_test, fused_fully_connected_layer) {
  fully_connected_layer_test(32, 64, 32);
  fully_connected_layer_test(2048, 512, 16);
  fully_connected_layer_test(2048, 1024, 480);
  fully_connected_layer_test(2048, 512, 1024);
  fully_connected_layer_test(2048, 1024, 1024);
}
