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

#include "absl/hash/hash.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/util/resource_cache.h"

#ifdef __APPLE__
#include "mediapipe/gpu/pixel_buffer_pool_util.h"
#endif  // __APPLE__

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/cv_pixel_buffer_pool_wrapper.h"
#else
#include "mediapipe/gpu/gl_texture_buffer_pool.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

struct GpuSharedData;
class CvPixelBufferPoolWrapper;

class GpuBufferMultiPool {
 public:
  GpuBufferMultiPool() {}

  // Obtains a buffer. May either be reused or created anew.
  GpuBuffer GetBuffer(int width, int height,
                      GpuBufferFormat format = GpuBufferFormat::kBGRA32);

  void SetFlushPlatformCaches(std::function<void(void)> flush_platform_caches) {
    flush_platform_caches_ = flush_platform_caches;
  }

  // This class is not intended as part of the public api of this class. It is
  // public only because it is used as a map key type, and the map
  // implementation needs access to, e.g., the equality operator.
  struct BufferSpec {
    BufferSpec(int w, int h, mediapipe::GpuBufferFormat f)
        : width(w), height(h), format(f) {}

    template <typename H>
    friend H AbslHashValue(H h, const BufferSpec& spec) {
      return H::combine(std::move(h), spec.width, spec.height,
                        static_cast<uint32_t>(spec.format));
    }

    int width;
    int height;
    mediapipe::GpuBufferFormat format;
  };

 private:
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  using SimplePool = CvPixelBufferPoolWrapper;
#else
  using SimplePool = GlTextureBufferPool;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

  std::shared_ptr<SimplePool> MakeSimplePool(const BufferSpec& spec);
  // Requests a simple buffer pool for the given spec. This may return nullptr
  // if we have not yet reached a sufficient number of requests to allocate a
  // pool, in which case the caller should invoke GetBufferWithoutPool instead
  // of GetBufferFromSimplePool.
  std::shared_ptr<SimplePool> RequestPool(const BufferSpec& spec);
  GpuBuffer GetBufferFromSimplePool(BufferSpec spec, SimplePool& pool);
  GpuBuffer GetBufferWithoutPool(const BufferSpec& spec);

  absl::Mutex mutex_;
  mediapipe::ResourceCache<BufferSpec, std::shared_ptr<SimplePool>> cache_
      ABSL_GUARDED_BY(mutex_);
  // This is used to hook up the TextureCacheManager on Apple platforms.
  std::function<void(void)> flush_platform_caches_;
};

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
