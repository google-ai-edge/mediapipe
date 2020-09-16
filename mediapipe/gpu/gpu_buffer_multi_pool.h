// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This class lets calculators allocate GpuBuffers of various sizes, caching
// and reusing them as needed. It does so by automatically creating and using
// platform-specific buffer pools for the requested sizes.
//
// This class is not meant to be used directly by calculators, but is instead
// used by GlCalculatorHelper to allocate buffers.

#ifndef MEDIAPIPE_GPU_GPU_BUFFER_MULTI_POOL_H_
#define MEDIAPIPE_GPU_GPU_BUFFER_MULTI_POOL_H_

#include <deque>
#include <limits>
#include <unordered_map>

#include "absl/synchronization/mutex.h"
#include "mediapipe/gpu/gpu_buffer.h"

#ifdef __APPLE__
#include "mediapipe/gpu/pixel_buffer_pool_util.h"
#endif  // __APPLE__

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_texture_buffer_pool.h"
#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

struct GpuSharedData;
class CvPixelBufferPoolWrapper;

class GpuBufferMultiPool {
 public:
  GpuBufferMultiPool() {}
  explicit GpuBufferMultiPool(void* ignored) {}
  ~GpuBufferMultiPool();

  // Obtains a buffer. May either be reused or created anew.
  GpuBuffer GetBuffer(int width, int height,
                      GpuBufferFormat format = GpuBufferFormat::kBGRA32);

#ifdef __APPLE__
  // TODO: add tests for the texture cache registration.

  // Inform the pool of a cache that should be flushed when it is low on
  // reusable buffers.
  void RegisterTextureCache(CVTextureCacheType cache);

  // Remove a texture cache from the list of caches to be flushed.
  void UnregisterTextureCache(CVTextureCacheType cache);

  void FlushTextureCaches();
#endif  // defined(__APPLE__)

  // This generates a "rol" instruction with both Clang and GCC.
  inline static std::size_t RotateLeft(std::size_t x, int n) {
    return (x << n) | (x >> (std::numeric_limits<size_t>::digits - n));
  }

  struct BufferSpec {
    BufferSpec(int w, int h, mediapipe::GpuBufferFormat f)
        : width(w), height(h), format(f) {}
    int width;
    int height;
    mediapipe::GpuBufferFormat format;
  };

  struct BufferSpecHash {
    std::size_t operator()(const BufferSpec& spec) const {
      // Width and height are expected to be smaller than half the width of
      // size_t. We can combine them into a single integer, and then use
      // std::hash, which is what go/hashing recommends for hashing numbers.
      constexpr int kWidth = std::numeric_limits<size_t>::digits;
      return std::hash<std::size_t>{}(
          spec.width ^ RotateLeft(spec.height, kWidth / 2) ^
          RotateLeft(static_cast<uint32_t>(spec.format), kWidth / 4));
    }
  };

 private:
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  using SimplePool = std::shared_ptr<CvPixelBufferPoolWrapper>;
#else
  using SimplePool = std::shared_ptr<GlTextureBufferPool>;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

  struct Entry {
    Entry(const BufferSpec& spec) : spec(spec) {}
    Entry* prev = nullptr;
    Entry* next = nullptr;
    BufferSpec spec;
    int request_count = 0;
    SimplePool pool;
  };

  // Unlike std::list, this is an intrusive list, meaning that the prev and next
  // pointers live inside the element. Apart from not requiring an extra
  // allocation, this means that once we look up an entry by key in the pools_
  // map we do not need to look it up separately in the list.
  //
  class EntryList {
   public:
    void Prepend(Entry* entry);
    void Append(Entry* entry);
    void Remove(Entry* entry);
    void InsertAfter(Entry* entry, Entry* after);

    Entry* head() { return head_; }
    Entry* tail() { return tail_; }
    size_t size() { return size_; }

   private:
    Entry* head_ = nullptr;
    Entry* tail_ = nullptr;
    size_t size_ = 0;
  };

  SimplePool MakeSimplePool(const BufferSpec& spec);
  // Requests a simple buffer pool for the given spec. This may return nullptr
  // if we have not yet reached a sufficient number of requests to allocate a
  // pool, in which case the caller should invoke GetBufferWithoutPool instead
  // of GetBufferFromSimplePool.
  SimplePool RequestPool(const BufferSpec& key);
  GpuBuffer GetBufferFromSimplePool(BufferSpec spec, const SimplePool& pool);
  GpuBuffer GetBufferWithoutPool(const BufferSpec& spec);
  void Evict(std::vector<SimplePool>* evicted)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  absl::Mutex mutex_;
  std::unordered_map<BufferSpec, Entry, BufferSpecHash> pools_
      ABSL_GUARDED_BY(mutex_);
  EntryList entry_list_ ABSL_GUARDED_BY(mutex_);
  int total_request_count_ = 0;

#ifdef __APPLE__
  // Texture caches used with this pool.
  std::vector<CFHolder<CVTextureCacheType>> texture_caches_ GUARDED_BY(mutex_);
#endif  // defined(__APPLE__)
};

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
class CvPixelBufferPoolWrapper {
 public:
  CvPixelBufferPoolWrapper(const GpuBufferMultiPool::BufferSpec& spec,
                           CFTimeInterval maxAge);
  GpuBuffer GetBuffer(std::function<void(void)> flush);

  int GetBufferCount() const { return count_; }
  std::string GetDebugString() const;

  void Flush();

 private:
  CFHolder<CVPixelBufferPoolRef> pool_;
  int count_ = 0;
};
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

// BufferSpec equality operators
inline bool operator==(const GpuBufferMultiPool::BufferSpec& lhs,
                       const GpuBufferMultiPool::BufferSpec& rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.format == rhs.format;
}
inline bool operator!=(const GpuBufferMultiPool::BufferSpec& lhs,
                       const GpuBufferMultiPool::BufferSpec& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_MULTI_POOL_H_
