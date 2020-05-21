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

struct BufferSpec {
  BufferSpec(int w, int h, GpuBufferFormat f)
      : width(w), height(h), format(f) {}
  int width;
  int height;
  GpuBufferFormat format;
};

inline bool operator==(const BufferSpec& lhs, const BufferSpec& rhs) {
  return lhs.width == rhs.width && lhs.height == rhs.height &&
         lhs.format == rhs.format;
}
inline bool operator!=(const BufferSpec& lhs, const BufferSpec& rhs) {
  return !operator==(lhs, rhs);
}

// This generates a "rol" instruction with both Clang and GCC.
static inline std::size_t RotateLeft(std::size_t x, int n) {
  return (x << n) | (x >> (std::numeric_limits<size_t>::digits - n));
}

struct BufferSpecHash {
  std::size_t operator()(const mediapipe::BufferSpec& spec) const {
    // Width and height are expected to be smaller than half the width of
    // size_t. We can combine them into a single integer, and then use
    // std::hash, which is what go/hashing recommends for hashing numbers.
    constexpr int kWidth = std::numeric_limits<size_t>::digits;
    return std::hash<std::size_t>{}(
        spec.width ^ RotateLeft(spec.height, kWidth / 2) ^
        RotateLeft(static_cast<uint32_t>(spec.format), kWidth / 4));
  }
};

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
#endif  // defined(__APPLE__)

 private:
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  typedef CFHolder<CVPixelBufferPoolRef> SimplePool;
#else
  typedef std::shared_ptr<GlTextureBufferPool> SimplePool;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

  SimplePool MakeSimplePool(const BufferSpec& spec);
  SimplePool GetSimplePool(const BufferSpec& key);
  GpuBuffer GetBufferFromSimplePool(BufferSpec spec, const SimplePool& pool);

  absl::Mutex mutex_;
  std::unordered_map<BufferSpec, SimplePool, BufferSpecHash> pools_
      ABSL_GUARDED_BY(mutex_);
  // A queue of BufferSpecs to keep track of the age of each BufferSpec added to
  // the pool.
  std::deque<BufferSpec> buffer_specs_;

#ifdef __APPLE__
  // Texture caches used with this pool.
  std::vector<CFHolder<CVTextureCacheType>> texture_caches_ GUARDED_BY(mutex_);
#endif  // defined(__APPLE__)
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GPU_BUFFER_MULTI_POOL_H_
