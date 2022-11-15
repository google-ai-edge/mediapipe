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

#include "mediapipe/gpu/gpu_buffer_multi_pool.h"

#include <tuple>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "CoreFoundation/CFBase.h"
#include "mediapipe/objc/CFHolder.h"
#include "mediapipe/objc/util.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

// Keep this many buffers allocated for a given frame size.
static constexpr int kKeepCount = 2;
// The maximum size of the GpuBufferMultiPool. When the limit is reached, the
// oldest BufferSpec will be dropped.
static constexpr int kMaxPoolCount = 10;
// Time in seconds after which an inactive buffer can be dropped from the pool.
// Currently only used with CVPixelBufferPool.
static constexpr float kMaxInactiveBufferAge = 0.25;
// Skip allocating a buffer pool until at least this many requests have been
// made for a given BufferSpec.
static constexpr int kMinRequestsBeforePool = 2;
// Do a deeper flush every this many requests.
static constexpr int kRequestCountScrubInterval = 50;

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

std::shared_ptr<GpuBufferMultiPool::SimplePool>
GpuBufferMultiPool::MakeSimplePool(const GpuBufferMultiPool::BufferSpec& spec) {
  return std::make_shared<CvPixelBufferPoolWrapper>(
      spec.width, spec.height, spec.format, kMaxInactiveBufferAge,
      flush_platform_caches_);
}

#else

std::shared_ptr<GpuBufferMultiPool::SimplePool>
GpuBufferMultiPool::MakeSimplePool(const BufferSpec& spec) {
  return GlTextureBufferPool::Create(spec.width, spec.height, spec.format,
                                     kKeepCount);
}

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

std::shared_ptr<GpuBufferMultiPool::SimplePool> GpuBufferMultiPool::RequestPool(
    const BufferSpec& spec) {
  std::shared_ptr<SimplePool> pool;
  std::vector<std::shared_ptr<SimplePool>> evicted;
  {
    absl::MutexLock lock(&mutex_);
    pool =
        cache_.Lookup(spec, [this](const BufferSpec& spec, int request_count) {
          return (request_count >= kMinRequestsBeforePool)
                     ? MakeSimplePool(spec)
                     : nullptr;
        });
    evicted = cache_.Evict(kMaxPoolCount, kRequestCountScrubInterval);
  }
  // Evicted pools, and their buffers, will be released without holding the
  // lock.
  return pool;
}

GpuBuffer GpuBufferMultiPool::GetBuffer(int width, int height,
                                        GpuBufferFormat format) {
  BufferSpec key(width, height, format);
  std::shared_ptr<SimplePool> pool = RequestPool(key);
  if (pool) {
    // Note: we release our multipool lock before accessing the simple pool.
    return GetBufferFromSimplePool(key, *pool);
  } else {
    return GetBufferWithoutPool(key);
  }
}

GpuBuffer GpuBufferMultiPool::GetBufferFromSimplePool(
    BufferSpec spec, GpuBufferMultiPool::SimplePool& pool) {
  return GpuBuffer(pool.GetBuffer());
}

GpuBuffer GpuBufferMultiPool::GetBufferWithoutPool(const BufferSpec& spec) {
  return GpuBuffer(SimplePool::CreateBufferWithoutPool(spec.width, spec.height,
                                                       spec.format));
}

}  // namespace mediapipe
