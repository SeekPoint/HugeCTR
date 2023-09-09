/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CONCURRENT_UNORDERED_MAP_CUH
#define CONCURRENT_UNORDERED_MAP_CUH

#include <thrust/pair.h>

#include <cassert>
#include <iostream>
#include <iterator>
#include <type_traits>

#include "hash_functions.cuh"
#include "managed.cuh"
#include "managed_allocator.cuh"

// TODO: replace this with CUDA_TRY and propagate the error
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                       \
  {                                                                                              \
    cudaError_t cudaStatus = call;                                                               \
    if (cudaSuccess != cudaStatus) {                                                             \
      fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);            \
      exit(1);                                                                                   \
    }                                                                                            \
  }
#endif

// TODO: can we do this more efficiently?
__inline__ __device__ int8_t atomicCAS(int8_t* address, int8_t compare, int8_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 3));
  int32_t int_val = (int32_t)val << (((size_t)address & 3) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 3) * 8);
  return (int8_t)atomicCAS(base_address, int_comp, int_val);
}

// TODO: can we do this more efficiently?
__inline__ __device__ int16_t atomicCAS(int16_t* address, int16_t compare, int16_t val) {
  int32_t* base_address = (int32_t*)((char*)address - ((size_t)address & 2));
  int32_t int_val = (int32_t)val << (((size_t)address & 2) * 8);
  int32_t int_comp = (int32_t)compare << (((size_t)address & 2) * 8);
  return (int16_t)atomicCAS(base_address, int_comp, int_val);
}

__inline__ __device__ int64_t atomicCAS(int64_t* address, int64_t compare, int64_t val) {
  return (int64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                            (unsigned long long)val);
}

// __inline__ __device__ uint64_t atomicCAS(uint64_t* address, uint64_t compare, uint64_t val) {
//   return (uint64_t)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
//                              (unsigned long long)val);
// }

__inline__ __device__ long long int atomicCAS(long long int* address, long long int compare,
                                              long long int val) {
  return (long long int)atomicCAS((unsigned long long*)address, (unsigned long long)compare,
                                  (unsigned long long)val);
}

__inline__ __device__ double atomicCAS(double* address, double compare, double val) {
  return __longlong_as_double(atomicCAS((unsigned long long int*)address,
                                        __double_as_longlong(compare), __double_as_longlong(val)));
}

__inline__ __device__ float atomicCAS(float* address, float compare, float val) {
  return __int_as_float(atomicCAS((int*)address, __float_as_int(compare), __float_as_int(val)));
}

__inline__ __device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  return (int64_t)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

__inline__ __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val) {
  return (uint64_t)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

template <typename pair_type>
__forceinline__ __device__ pair_type load_pair_vectorized(const pair_type* __restrict__ const ptr) {
  if (sizeof(uint4) == sizeof(pair_type)) {
    union pair_type2vec_type {
      uint4 vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0, 0, 0, 0};
    converter.vec_val = *reinterpret_cast<const uint4*>(ptr);
    return converter.pair_val;
  } else if (sizeof(uint2) == sizeof(pair_type)) {
    union pair_type2vec_type {
      uint2 vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0, 0};
    converter.vec_val = *reinterpret_cast<const uint2*>(ptr);
    return converter.pair_val;
  } else if (sizeof(int) == sizeof(pair_type)) {
    union pair_type2vec_type {
      int vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0};
    converter.vec_val = *reinterpret_cast<const int*>(ptr);
    return converter.pair_val;
  } else if (sizeof(short) == sizeof(pair_type)) {
    union pair_type2vec_type {
      short vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0};
    converter.vec_val = *reinterpret_cast<const short*>(ptr);
    return converter.pair_val;
  } else {
    return *ptr;
  }
}

template <typename pair_type>
__forceinline__ __device__ void store_pair_vectorized(pair_type* __restrict__ const ptr,
                                                      const pair_type val) {
  if (sizeof(uint4) == sizeof(pair_type)) {
    union pair_type2vec_type {
      uint4 vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0, 0, 0, 0};
    converter.pair_val = val;
    *reinterpret_cast<uint4*>(ptr) = converter.vec_val;
  } else if (sizeof(uint2) == sizeof(pair_type)) {
    union pair_type2vec_type {
      uint2 vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0, 0};
    converter.pair_val = val;
    *reinterpret_cast<uint2*>(ptr) = converter.vec_val;
  } else if (sizeof(int) == sizeof(pair_type)) {
    union pair_type2vec_type {
      int vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0};
    converter.pair_val = val;
    *reinterpret_cast<int*>(ptr) = converter.vec_val;
  } else if (sizeof(short) == sizeof(pair_type)) {
    union pair_type2vec_type {
      short vec_val;
      pair_type pair_val;
    };
    pair_type2vec_type converter = {0};
    converter.pair_val = val;
    *reinterpret_cast<short*>(ptr) = converter.vec_val;
  } else {
    *ptr = val;
  }
}

template <typename value_type, typename size_type, typename key_type, typename elem_type>
__global__ void init_hashtbl(  // Init every entry of the table with <unused_key, unused_value> pair
    value_type* __restrict__ const hashtbl_values, const size_type n, const key_type key_val,
    const elem_type elem_val) {
  const size_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    store_pair_vectorized(
        hashtbl_values + idx,
        thrust::make_pair(key_val, elem_val));  // Simply store every element a <K, V> pair
  }
}

template <typename T>
struct equal_to {
  using result_type = bool;
  using first_argument_type = T;
  using second_argument_type = T;
  __forceinline__ __host__ __device__ constexpr bool operator()(
      const first_argument_type& lhs, const second_argument_type& rhs) const {
    return lhs == rhs;
  }
};

template <typename Iterator>
class cycle_iterator_adapter {
 public:
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;
  using pointer = typename std::iterator_traits<Iterator>::pointer;
  using reference = typename std::iterator_traits<Iterator>::reference;
  using iterator_type = Iterator;

  cycle_iterator_adapter() = delete;

  __host__ __device__ explicit cycle_iterator_adapter(const iterator_type& begin,
                                                      const iterator_type& end,
                                                      const iterator_type& current)
      : m_begin(begin), m_end(end), m_current(current) {}

  __host__ __device__ cycle_iterator_adapter& operator++() {
    if (m_end == (m_current + 1))
      m_current = m_begin;
    else
      ++m_current;
    return *this;
  }

  __host__ __device__ const cycle_iterator_adapter& operator++() const {
    if (m_end == (m_current + 1))
      m_current = m_begin;
    else
      ++m_current;
    return *this;
  }

  __host__ __device__ cycle_iterator_adapter& operator++(int) {
    cycle_iterator_adapter<iterator_type> old(m_begin, m_end, m_current);
    if (m_end == (m_current + 1))
      m_current = m_begin;
    else
      ++m_current;
    return old;
  }

  __host__ __device__ const cycle_iterator_adapter& operator++(int) const {
    cycle_iterator_adapter<iterator_type> old(m_begin, m_end, m_current);
    if (m_end == (m_current + 1))
      m_current = m_begin;
    else
      ++m_current;
    return old;
  }

  __host__ __device__ bool equal(const cycle_iterator_adapter<iterator_type>& other) const {
    return m_current == other.m_current && m_begin == other.m_begin && m_end == other.m_end;
  }

  __host__ __device__ reference& operator*() { return *m_current; }

  __host__ __device__ const reference& operator*() const { return *m_current; }

  __host__ __device__ const pointer operator->() const { return m_current.operator->(); }

  __host__ __device__ pointer operator->() { return m_current; }

  __host__ __device__ iterator_type getter() const { return m_current; }

 private:
  iterator_type m_current;
  iterator_type m_begin;
  iterator_type m_end;
};

template <class T>
__host__ __device__ bool operator==(const cycle_iterator_adapter<T>& lhs,
                                    const cycle_iterator_adapter<T>& rhs) {
  return lhs.equal(rhs);
}

template <class T>
__host__ __device__ bool operator!=(const cycle_iterator_adapter<T>& lhs,
                                    const cycle_iterator_adapter<T>& rhs) {
  return !lhs.equal(rhs);
}

/**
 * Does support concurrent insert, but not concurrent insert and probping.
 *
 * TODO:
 *  - add constructor that takes pointer to hash_table to avoid allocations
 *  - extend interface to accept streams

 3.4 concurrent_unordered_map
concurrent_unordered_map 定义在 HugeCTR/include/hashtable/cudf/concurrent_unordered_map.cuh。

这是位于显存中的map。从其注释可知，其支持并发插入，但是不支持同时insert和probping。
 结合HugeCTR看，hugeCTR是同步训练，pull操作只会调用 get，push操作只会调用insert，不存在同时insert和probping，所以满足需求。
 */
template <typename Key, typename Element, Key unused_key, typename Hasher = default_hash<Key>,
          typename Equality = equal_to<Key>,
          typename Allocator = managed_allocator<thrust::pair<Key, Element>>,
          bool count_collisions = false>
class concurrent_unordered_map : public managed {
 public:
  using size_type = size_t;
  using hasher = Hasher;
  using key_equal = Equality;
  using allocator_type = Allocator;
  using key_type = Key;
  using value_type = thrust::pair<Key, Element>;
  using mapped_type = Element;
  using iterator = cycle_iterator_adapter<value_type*>;
  using const_iterator = const cycle_iterator_adapter<value_type*>;

 private:
  union pair2longlong {
    unsigned long long int longlong;
    value_type pair;
  };

 public:
  concurrent_unordered_map(const concurrent_unordered_map&) = delete;
  concurrent_unordered_map& operator=(const concurrent_unordered_map&) = delete;
  explicit concurrent_unordered_map(size_type n, const mapped_type unused_element,
                                    const Hasher& hf = hasher(), const Equality& eql = key_equal(),
                                    const allocator_type& a = allocator_type())
      : m_hf(hf),
        m_equal(eql),
        m_allocator(a),
        m_hashtbl_size(n),
        m_hashtbl_capacity(n),
        m_collisions(0),
        m_unused_element(unused_element) {  // allocate the raw data of hash table:
                                            // m_hashtbl_values,pre-alloc it on current GPU if UM.
    m_hashtbl_values = m_allocator.allocate(m_hashtbl_capacity);
    constexpr int block_size = 128;
    {
      cudaPointerAttributes hashtbl_values_ptr_attributes;
      cudaError_t status =
          cudaPointerGetAttributes(&hashtbl_values_ptr_attributes, m_hashtbl_values);

#if CUDART_VERSION >= 10000
      if (cudaSuccess == status && hashtbl_values_ptr_attributes.type == cudaMemoryTypeManaged)
#else
      if (cudaSuccess == status && hashtbl_values_ptr_attributes.isManaged)
#endif
      {
        int dev_id = 0;
        CUDA_RT_CALL(cudaGetDevice(&dev_id));
        CUDA_RT_CALL(
            cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size * sizeof(value_type), dev_id, 0));
      }
    }
    // Initialize kernel, set all entry to unused <K,V>
    init_hashtbl<<<((m_hashtbl_size - 1) / block_size) + 1, block_size>>>(
        m_hashtbl_values, m_hashtbl_size, unused_key, m_unused_element);
    // CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL(cudaStreamSynchronize(0));
    CUDA_RT_CALL(cudaGetLastError());
  }

  ~concurrent_unordered_map() { m_allocator.deallocate(m_hashtbl_values, m_hashtbl_capacity); }

  __host__ __device__ iterator begin() {
    return iterator(m_hashtbl_values, m_hashtbl_values + m_hashtbl_size, m_hashtbl_values);
  }
  __host__ __device__ const_iterator begin() const {
    return const_iterator(m_hashtbl_values, m_hashtbl_values + m_hashtbl_size, m_hashtbl_values);
  }
  __host__ __device__ iterator end() {
    return iterator(m_hashtbl_values, m_hashtbl_values + m_hashtbl_size,
                    m_hashtbl_values + m_hashtbl_size);
  }
  __host__ __device__ const_iterator end() const {
    return const_iterator(m_hashtbl_values, m_hashtbl_values + m_hashtbl_size,
                          m_hashtbl_values + m_hashtbl_size);
  }
  __host__ __device__ size_type size() const { return m_hashtbl_size; }
  __host__ __device__ value_type* data() const { return m_hashtbl_values; }

  __forceinline__ static constexpr __host__ __device__ key_type get_unused_key() {
    return unused_key;
  }

  // Generic update of a hash table value for any aggregator
  template <typename aggregation_type>
  __forceinline__ __device__ void update_existing_value(mapped_type& existing_value,
                                                        value_type const& insert_pair,
                                                        aggregation_type) {
    // update without CAS
    existing_value = insert_pair.second;
  }

  __forceinline__ __device__ void accum_existing_value_atomic(mapped_type& existing_value,
                                                              value_type const& accum_pair) {
    // update with CAS
    // existing_value = insert_pair.second;
    int num_element = sizeof(existing_value.data) / sizeof(*(existing_value.data));
    const mapped_type& accumulator = accum_pair.second;

    for (int i = 0; i < num_element; i++) {
      atomicAdd(existing_value.data + i, accumulator.data[i]);
    }

    // atomicAdd(&existing_value, double val)
  }

  // TODO Overload atomicAdd for 1 byte and 2 byte types, until then, overload specifically for the
  // types where atomicAdd already has an overload. Otherwise the generic update_existing_value will
  // be used. Specialization for COUNT aggregator
  /*
  __forceinline__ __host__ __device__
  void update_existing_value(mapped_type & existing_value, value_type const & insert_pair,
  count_op<int32_t> op)
  {
    atomicAdd(&existing_value, static_cast<mapped_type>(1));
  }
  // Specialization for COUNT aggregator
  __forceinline__ __host__ __device__
  void update_existing_value(mapped_type & existing_value, value_type const & insert_pair,
  count_op<int64_t> op)
  {
    atomicAdd(&existing_value, static_cast<mapped_type>(1));
  }
  // Specialization for COUNT aggregator
  __forceinline__ __host__ __device__
  void update_existing_value(mapped_type & existing_value, value_type const & insert_pair,
  count_op<float> op)
  {
    atomicAdd(&existing_value, static_cast<mapped_type>(1));
  }
  // Specialization for COUNT aggregator
  __forceinline__ __host__ __device__
  void update_existing_value(mapped_type & existing_value, value_type const & insert_pair,
  count_op<double> op)
  {
    atomicAdd(&existing_value, static_cast<mapped_type>(1));
  }
  */

  /* --------------------------------------------------------------------------*/
  /**
   * @Synopsis  Inserts a new (key, value) pair. If the key already exists in the map
                an aggregation operation is performed with the new value and existing value.
                E.g., if the aggregation operation is 'max', then the maximum is computed
                between the new value and existing value and the result is stored in the map.
   *
   * @Param[in] x The new (key, value) pair to insert
   * @Param[in] op The aggregation operation to perform
   * @Param[in] keys_equal An optional functor for comparing two keys
   * @Param[in] precomputed_hash Indicates if a precomputed hash value is being passed in to use
   * to determine the write location of the new key
   * @Param[in] precomputed_hash_value The precomputed hash value
   * @tparam aggregation_type A functor for a binary operation that performs the aggregation
   * @tparam comparison_type A functor for comparing two keys
   *
   * @Returns An iterator to the newly inserted key,value pair
   */
  /* ----------------------------------------------------------------------------*/
  template <typename aggregation_type, class comparison_type = key_equal,
            typename hash_value_type = typename Hasher::result_type>
  __forceinline__ __device__ iterator insert(const value_type& x, aggregation_type op,
                                             comparison_type keys_equal = key_equal(),
                                             bool precomputed_hash = false,
                                             hash_value_type precomputed_hash_value = 0) {
    const size_type hashtbl_size = m_hashtbl_size;
    value_type* hashtbl_values = m_hashtbl_values;

    hash_value_type hash_value{0};

    // If a precomputed hash value has been passed in, then use it to determine
    // the write location of the new key
    if (true == precomputed_hash) {
      hash_value = precomputed_hash_value;
    }
    // Otherwise, compute the hash value from the new key
    else {
      hash_value = m_hf(x.first);
    }

    size_type current_index = hash_value % hashtbl_size;
    value_type* current_hash_bucket = &(hashtbl_values[current_index]);

    const key_type insert_key = x.first;

    bool insert_success = false;

    size_type counter = 0;
    while (false == insert_success) {
      if (counter++ >= hashtbl_size) {
        return end();
      }

      key_type& existing_key = current_hash_bucket->first;
      mapped_type& existing_value = current_hash_bucket->second;

      // Try and set the existing_key for the current hash bucket to insert_key
      const key_type old_key = atomicCAS(&existing_key, unused_key, insert_key);

      // If old_key == unused_key, the current hash bucket was empty
      // and existing_key was updated to insert_key by the atomicCAS.
      // If old_key == insert_key, this key has already been inserted.
      // In either case, perform the atomic aggregation of existing_value and insert_value
      // Because the hash table is initialized with the identity value of the aggregation
      // operation, it is safe to perform the operation when the existing_value still
      // has its initial value
      // TODO: Use template specialization to make use of native atomic functions
      // TODO: How to handle data types less than 32 bits?
      if (keys_equal(unused_key, old_key) || keys_equal(insert_key, old_key)) {
        update_existing_value(existing_value, x, op);

        insert_success = true;
      }

      current_index = (current_index + 1) % hashtbl_size;
      current_hash_bucket = &(hashtbl_values[current_index]);
    }

    return iterator(m_hashtbl_values, m_hashtbl_values + hashtbl_size, current_hash_bucket);
  }

  /* This function is not currently implemented
  __forceinline__
  __host__ __device__ iterator insert(const value_type& x)
  {
      const size_type hashtbl_size    = m_hashtbl_size;
      value_type* hashtbl_values      = m_hashtbl_values;
      const size_type key_hash        = m_hf( x.first );
      size_type hash_tbl_idx          = key_hash%hashtbl_size;

      value_type* it = 0;

      while (0 == it) {
          value_type* tmp_it = hashtbl_values + hash_tbl_idx;
#ifdef __CUDA_ARCH__
          if ( std::numeric_limits<key_type>::is_integer &&
std::numeric_limits<mapped_type>::is_integer && sizeof(unsigned long long int) == sizeof(value_type)
)
          {
              pair2longlong converter = {0ull};
              converter.pair = thrust::make_pair( unused_key, m_unused_element );
              const unsigned long long int unused = converter.longlong;
              converter.pair = x;
              const unsigned long long int value = converter.longlong;
              const unsigned long long int old_val = atomicCAS( reinterpret_cast<unsigned long long
int*>(tmp_it), unused, value ); if ( old_val == unused ) { it = tmp_it;
              }
              else if ( count_collisions )
              {
                  atomicAdd( &m_collisions, 1 );
              }
          } else {
              const key_type old_key = atomicCAS( &(tmp_it->first), unused_key, x.first );
              if ( m_equal( unused_key, old_key ) ) {
                  (m_hashtbl_values+hash_tbl_idx)->second = x.second;
                  it = tmp_it;
              }
              else if ( count_collisions )
              {
                  atomicAdd( &m_collisions, 1 );
              }
          }
#else

          #pragma omp critical
          {
              if ( m_equal( unused_key, tmp_it->first ) ) {
                  hashtbl_values[hash_tbl_idx] = thrust::make_pair( x.first, x.second );
                  it = tmp_it;
              }
          }
#endif
          hash_tbl_idx = (hash_tbl_idx+1)%hashtbl_size;
      }

      return iterator( m_hashtbl_values,m_hashtbl_values+hashtbl_size,it);
  }
  */
  // __forceinline__ 的意思是编译为内联函数
  // __host__ __device__ 表示是此函数同时为主机和设备编译
  __forceinline__ __host__ __device__ const_iterator find(const key_type& k) const {
    // 对key进行hash操作
    size_type key_hash = m_hf(k);
    // 进而得到table的相应index
    size_type hash_tbl_idx = key_hash % m_hashtbl_size;

    value_type* begin_ptr = 0;

    size_type counter = 0;
    while (0 == begin_ptr) {
      value_type* tmp_ptr = m_hashtbl_values + hash_tbl_idx;
      const key_type tmp_val = tmp_ptr->first;
      // 找到key，跳出
      if (m_equal(k, tmp_val)) {
        begin_ptr = tmp_ptr;
        break;
      }
      // key的位置是空，或者在table之内没有找到
      if (m_equal(unused_key, tmp_val) || counter > m_hashtbl_size) {
        begin_ptr = m_hashtbl_values + m_hashtbl_size;
        break;
      }
      hash_tbl_idx = (hash_tbl_idx + 1) % m_hashtbl_size;
      ++counter;
    }

    return const_iterator(m_hashtbl_values, m_hashtbl_values + m_hashtbl_size, begin_ptr);
  }
/*
  3.4.2 insert
  插入操作我们就看看之前的 get_insert。
  hash_table.get_insert(hash_key.get_ptr(), hash_value_index.get_ptr(), nnz, stream);
  就是以 csr 部分信息作为 hash key，来获得一个低维嵌入表之中的index，在 hash_value_index之中返回。我们首先看一个CSR示例。

      * For example data:
      *   3356
      *   667
      *   588
      * Will be convert to the form of:
      * row offset: 0,1,2,3
      * value: 3356,667,588,3

我们就是使用 3356 作为 hash_key，获取 3356 对应的 hash_value_index，如果能找到就返回，找不到就插入一个构建的value，然后这个 value 会返回给 hash_value_index。

但是这里有几个绕的地方，因为 HashTable内部也分桶，也有自己的key，hash_value，容易和其他数据结构弄混。具体逻辑是：
      传入一个数字 3356（CSR格式相关），还有一个 value_counter，就是目前 hash_value_index 的数值。
      先 hash_value = m_hf(3356)。
      用 current_index = hash_value % hashtbl_size 找到 m_hashtbl_values 之中的位置。
      用 current_hash_bucket = &(hashtbl_values[current_index]) 这找到了一个bucket。
      key_type& existing_key = current_hash_bucket->first，这个才是 hash table key
      volatile mapped_type& existing_value = current_hash_bucket->second，这个才是我们最终需要的 table value。如果没有，就递增传入的 value_counter。
所以，CSR 3356 是一个one-hot 的index，它对应了embeding表的一个index，但是因为没有那么大的embedding，所以后面会构建一个小数据结构（低维矩阵） hash_value，
传入的 value_counter 就是这个 hash_value的index，value_counter 是递增的，因为 hash_value 的行号就是递增的。

比如一共有1亿个单词，3356表示第3356个单词。如果想表示 3356，667，588 这三个位置在这一亿个单词是有效的，
最笨的办法是弄个1亿长度数组，把3356，667，588这三个位置设置为 1，其他位置设置为0，
但是这样太占据空间且没有意义。如果想省空间，就弄一个hash函数 m_hf，假如是选取最高位数为 value，则得到：
  m_hf(3356)=3
  m_hf(667)=6
  m_hf(588)=5
3，5，6 就是内部的 hash_value，叫做 hash_value（对应下面代码），对应的内部存储数组叫做 hashtbl_values。
再梳理一下：3356是哈希表的key，3 是哈希表的value，但是因为分桶了，所以在哈希表内部是放置在 hashtbl_values 之中。
                    hashtbl_values[3] = 1，hashtbl_values[6] = 2, hashtbl_values[5] =3
于是 1，2，3 就是我们外部想得到的 3356, 667, 588 对应的数据，就是低维矩阵的 row offset，对应下面代码就是 existing_value。
 简化版本的逻辑如下:
       005-002.jpg
具体代码如下：

具体逻辑演进如下： 006-006.jpg

// __forceinline__ 的意思是编译为内联函数
 // __host__ __device__ 表示是此函数同时为主机和设备编译
  */
  template <typename aggregation_type, typename counter_type, class comparison_type = key_equal,
            typename hash_value_type = typename Hasher::result_type>
  __forceinline__ __device__ iterator get_insert(const key_type& k, aggregation_type op,
                                                 counter_type* value_counter,
                                                 comparison_type keys_equal = key_equal(),
                                                 bool precomputed_hash = false,
                                                 hash_value_type precomputed_hash_value = 0) {
    const size_type hashtbl_size = m_hashtbl_size;
    value_type* hashtbl_values = m_hashtbl_values;

    hash_value_type hash_value{0};

    // If a precomputed hash value has been passed in, then use it to determine
    // the write location of the new key
    if (true == precomputed_hash) {
      hash_value = precomputed_hash_value;
    }
    // Otherwise, compute the hash value from the new key
    else {
      hash_value = m_hf(k);  // 3356作为key，得到了一个hash_value
    }

    size_type current_index = hash_value % hashtbl_size;  // 找到哪个位置
    value_type* current_hash_bucket = &(hashtbl_values[current_index]); // 找到该位置的bucket

    const key_type insert_key = k;

    bool insert_success = false;

    size_type counter = 0;
    while (false == insert_success) {
      // Situation %5: No slot: All slot in the hashtable is occupied by other key, both get and
      // insert fail. Return empty iterator
      // hash表已经满了
      if (counter++ >= hashtbl_size) {
        return end();
      }

      key_type& existing_key = current_hash_bucket->first; // 这个才是table key
      volatile mapped_type& existing_value = current_hash_bucket->second; // 这个才是table value
      // 如果 existing_key == unused_key时，则当前哈希位置为空，所以existing_key由atomicCAS更新为insert_key。
      // 如果 existing_key == insert_key时，这个位置已经被插入这个key了。
      // 在任何一种情况下，都要执行existing_value和insert_value的atomic聚合，因为哈希表是用聚合操作的标识值初始化的，
      // 所以在existing_value仍具有其初始值时，执行该操作是安全的
      // Try and set the existing_key for the current hash bucket to insert_key
      const key_type old_key = atomicCAS(&existing_key, unused_key, insert_key);

      // If old_key == unused_key, the current hash bucket was empty
      // and existing_key was updated to insert_key by the atomicCAS.
      // If old_key == insert_key, this key has already been inserted.
      // In either case, perform the atomic aggregation of existing_value and insert_value
      // Because the hash table is initialized with the identity value of the aggregation
      // operation, it is safe to perform the operation when the existing_value still
      // has its initial value
      // TODO: Use template specialization to make use of native atomic functions
      // TODO: How to handle data types less than 32 bits?

      // Situation #1: Empty slot: this key never exist in the table, ready to insert.
      if (keys_equal(unused_key, old_key)) { // 如果没有找到hash key
        // update_existing_value(existing_value, x, op);
        existing_value = (mapped_type)(atomicAdd(value_counter, 1)); // hash value 就递增
        break;

      }  // Situation #2+#3: Target slot: This slot is the slot for this key
      else if (keys_equal(insert_key, old_key)) {
        while (existing_value == m_unused_element) {
          // Situation #2: This slot is inserting by another CUDA thread and the value is not yet
          // ready, just wait
        }
        // Situation #3: This slot is already ready, get successfully and return (iterator of) the
        // value
        break;
      }
      // Situation 4: Wrong slot: This slot is occupied by other key, get fail, do nothing and
      // linear probing to next slot.

      // 此位置已经被其他key占了，只能向后遍历
      current_index = (current_index + 1) % hashtbl_size;
      current_hash_bucket = &(hashtbl_values[current_index]);
    }

    return iterator(m_hashtbl_values, m_hashtbl_values + hashtbl_size, current_hash_bucket);
  }

  int assign_async(const concurrent_unordered_map& other, cudaStream_t stream = 0) {
    m_collisions = other.m_collisions;
    if (other.m_hashtbl_size <= m_hashtbl_capacity) {
      m_hashtbl_size = other.m_hashtbl_size;
    } else {
      m_allocator.deallocate(m_hashtbl_values, m_hashtbl_capacity);
      m_hashtbl_capacity = other.m_hashtbl_size;
      m_hashtbl_size = other.m_hashtbl_size;

      m_hashtbl_values = m_allocator.allocate(m_hashtbl_capacity);
    }
    CUDA_RT_CALL(cudaMemcpyAsync(m_hashtbl_values, other.m_hashtbl_values,
                                 m_hashtbl_size * sizeof(value_type), cudaMemcpyDefault, stream));
    return 0;
  }

  void clear_async(cudaStream_t stream = 0) {
    constexpr int block_size = 128;
    init_hashtbl<<<((m_hashtbl_size - 1) / block_size) + 1, block_size, 0, stream>>>(
        m_hashtbl_values, m_hashtbl_size, unused_key, m_unused_element);
    if (count_collisions) m_collisions = 0;
  }

  unsigned long long get_num_collisions() const { return m_collisions; }

  void print() {
    for (size_type i = 0; i < m_hashtbl_size; ++i) {
      std::cout << i << ": " << m_hashtbl_values[i].first << "," << m_hashtbl_values[i].second
                << std::endl;
    }
  }

  int prefetch(const int dev_id, cudaStream_t stream = 0) {
    cudaPointerAttributes hashtbl_values_ptr_attributes;
    cudaError_t status = cudaPointerGetAttributes(&hashtbl_values_ptr_attributes, m_hashtbl_values);

#if CUDART_VERSION >= 10000
    if (cudaSuccess == status && hashtbl_values_ptr_attributes.type == cudaMemoryTypeManaged)
#else
    if (cudaSuccess == status && hashtbl_values_ptr_attributes.isManaged)
#endif
    {
      CUDA_RT_CALL(cudaMemPrefetchAsync(m_hashtbl_values, m_hashtbl_size * sizeof(value_type),
                                        dev_id, stream));
    }
    CUDA_RT_CALL(cudaMemPrefetchAsync(this, sizeof(*this), dev_id, stream));

    return 0;
  }

  template <class comparison_type = key_equal,
            typename hash_value_type = typename Hasher::result_type>
  __forceinline__ __device__ const_iterator accum(const value_type& x,
                                                  comparison_type keys_equal = key_equal(),
                                                  bool precomputed_hash = false,
                                                  hash_value_type precomputed_hash_value = 0) {
    const key_type& dst_key = x.first;
    auto it = find(dst_key);

    if (it == end()) {
      return it;
    }

    value_type* dst = it.getter();

    accum_existing_value_atomic(dst->second, x);

    return it;
  }

 private:
  const hasher m_hf;
  const key_equal m_equal;

  const mapped_type m_unused_element;

  allocator_type m_allocator;

  size_type m_hashtbl_size;
  size_type m_hashtbl_capacity;
  value_type* m_hashtbl_values; // 这个才是hash数据结构位置

  unsigned long long m_collisions;
};

#endif  // CONCURRENT_UNORDERED_MAP_CUH
