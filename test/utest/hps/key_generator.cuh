/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#include <core23/logger.hpp>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

template <typename key_type>
__global__ void gen_keys_kernel(key_type* lookup_keys, bool uniform, const float alpha,
                                float* random_numbers, const size_t lookup_length,
                                const size_t num_key_candidates) {
  register size_t tId = blockIdx.x * blockDim.x + threadIdx.x, tNum = gridDim.x * blockDim.x;
  if (uniform) {
#pragma unroll
    for (register size_t i = tId; i < lookup_length; i += tNum) {
      lookup_keys[i] = (key_type)((1 - random_numbers[i]) * num_key_candidates);
      // 1 - random_numbers[i] for turning (0, 1] to [0, 1) as generated by cuRand
    }
  } else {  // power-law
    const float gamma = 1 - alpha;
    const float pMax = __powf(num_key_candidates, gamma),
                pMin = 1.;  // assuming 1^gamma = 1 always hold
#pragma unroll
    for (register size_t i = tId; i < lookup_length; i += tNum) {
      lookup_keys[i] =
          (key_type)__powf((1 - random_numbers[i]) * (pMax - pMin) + pMin, 1. / gamma) - 1.0f;
    }
  }
}

namespace {
// Fast random number generator.
// https://en.wikipedia.org/wiki/Xorshift
__host__ __device__ __forceinline__ uint64_t xorshift64star(uint64_t x, int seed) {
  x += seed;
  x ^= x >> 12;  // a
  x ^= x << 25;  // b
  x ^= x >> 27;  // c
  return x * 0x2545F4914F6CDD1D;
}
}  // namespace

template <typename key_type>
class KeyGenerator {
 public:
  KeyGenerator(const size_t batch_size, const size_t num_hot, const float alpha,
               const size_t num_key_candidates, const size_t key_range, const int seed = -1)
      : batch_size_(batch_size),
        num_hot_(num_hot),
        alpha_(alpha),
        num_key_candidates_(num_key_candidates),
        key_range_(key_range) {
    HCTR_LIB_THROW(curandCreateGenerator(&cu_generator_, CURAND_RNG_PSEUDO_MT19937));
    h_keys_.resize(batch_size_ * num_hot_);
    HCTR_LIB_THROW(cudaMalloc(&d_keys_buffer_, sizeof(*d_keys_buffer_) * batch_size_ * num_hot_));
    HCTR_LIB_THROW(
        cudaMalloc(&d_random_numbers_, sizeof(*d_random_numbers_) * batch_size_ * num_hot_));

    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProps;
    HCTR_LIB_THROW(cudaGetDeviceProperties(&deviceProps, device_id));

    n_sm_ = deviceProps.multiProcessorCount;
    max_block_num_per_sm_ = deviceProps.maxBlocksPerMultiProcessor;
    if (seed == -1) {
      seed_ = (int)time(NULL);
    }
    return;
  }

  std::vector<key_type> get_next_batch() {
    HCTR_LIB_THROW(curandGenerateUniform(cu_generator_, d_random_numbers_, batch_size_ * num_hot_));
    bool use_uniform = alpha_ == 0.0f;
    gen_keys_kernel<<<n_sm_ * max_block_num_per_sm_, 32, 0>>>(
        d_keys_buffer_, use_uniform, alpha_, d_random_numbers_, batch_size_ * num_hot_,
        num_key_candidates_);
    HCTR_LIB_THROW(cudaMemcpy(h_keys_.data(), d_keys_buffer_,
                              sizeof(*d_keys_buffer_) * batch_size_ * num_hot_,
                              cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h_keys_.size(); i++) {
      h_keys_[i] = map_key(h_keys_[i]);
    }
    return h_keys_;
  }

  uint64_t map_key(uint64_t key) {
    // return xorshift64star(key, seed_) % key_range_;
    return key;
  }

  ~KeyGenerator() {
    cudaFree(d_keys_buffer_);
    cudaFree(d_random_numbers_);

    curandDestroyGenerator(cu_generator_);
  }

 private:
  float alpha_;
  size_t batch_size_;
  size_t num_hot_;
  size_t num_key_candidates_;
  size_t key_range_;
  curandGenerator_t cu_generator_;
  float* d_random_numbers_;
  key_type* d_keys_buffer_;
  std::vector<key_type> h_keys_;
  int n_sm_;
  int max_block_num_per_sm_;
  int seed_;
};