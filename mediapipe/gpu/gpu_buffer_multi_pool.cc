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

CvPixelBufferPoolWrapper::CvPixelBufferPoolWrapper(
    const GpuBufferMultiPool::BufferSpec& spec, CFTimeInterval maxAge) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(spec.format);
  CHECK_NE(cv_format, -1) << "unsupported pixel format";
  pool_ = MakeCFHolderAdopting(
      /* keep count is 0 because the age param keeps buffers around anyway */
      CreateCVPixelBufferPool(spec.width, spec.height, cv_format, 0, maxAge));
}

GpuBuffer CvPixelBufferPoolWrapper::GetBuffer(std::function<void(void)> flush) {
  CVPixelBufferRef buffer;
  int threshold = 1;
  NSMutableDictionary* auxAttributes =
      [NSMutableDictionary dictionaryWithCapacity:1];
  CVReturn err;
  bool tried_flushing = false;
  while (1) {
    auxAttributes[(id)kCVPixelBufferPoolAllocationThresholdKey] = @(threshold);
    err = CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(
        kCFAllocatorDefault, *pool_, (__bridge CFDictionaryRef)auxAttributes,
        &buffer);
    if (err != kCVReturnWouldExceedAllocationThreshold) break;
    if (flush && !tried_flushing) {
      // Call the flush function to potentially release old holds on buffers
      // and try again to create a pixel buffer.
      // This is used to flush CV texture caches, which may retain buffers until
      // flushed.
      flush();
      tried_flushing = true;
    } else {
      ++threshold;
    }
  }
  CHECK(!err) << "Error creating pixel buffer: " << err;
  count_ = threshold;
  return GpuBuffer(MakeCFHolderAdopting(buffer));
}

std::string CvPixelBufferPoolWrapper::GetDebugString() const {
  auto description = MakeCFHolderAdopting(CFCopyDescription(*pool_));
  return [(__bridge NSString*)*description UTF8String];
}

void CvPixelBufferPoolWrapper::Flush() { CVPixelBufferPoolFlush(*pool_, 0); }

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::MakeSimplePool(
    const GpuBufferMultiPool::BufferSpec& spec) {
  return std::make_shared<CvPixelBufferPoolWrapper>(spec,
                                                    kMaxInactiveBufferAge);
}

GpuBuffer GpuBufferMultiPool::GetBufferWithoutPool(const BufferSpec& spec) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(spec.format);
  CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err = CreateCVPixelBufferWithoutPool(spec.width, spec.height,
                                                cv_format, &buffer);
  CHECK(!err) << "Error creating pixel buffer: " << err;
  return GpuBuffer(MakeCFHolderAdopting(buffer));
}

void GpuBufferMultiPool::FlushTextureCaches() {
  absl::MutexLock lock(&mutex_);
  for (const auto& cache : texture_caches_) {
#if TARGET_OS_OSX
    CVOpenGLTextureCacheFlush(*cache, 0);
#else
    CVOpenGLESTextureCacheFlush(*cache, 0);
#endif  // TARGET_OS_OSX
  }
}

// Turning this on disables the pixel buffer pools when using the simulator.
// It is no longer necessary, since the helper code now supports non-contiguous
// buffers. We leave the code in for now for the sake of documentation.
#define FORCE_CONTIGUOUS_PIXEL_BUFFER_ON_IPHONE_SIMULATOR 0

GpuBuffer GpuBufferMultiPool::GetBufferFromSimplePool(
    BufferSpec spec, const GpuBufferMultiPool::SimplePool& pool) {
#if TARGET_IPHONE_SIMULATOR && FORCE_CONTIGUOUS_PIXEL_BUFFER_ON_IPHONE_SIMULATOR
  // On the simulator, syncing the texture with the pixelbuffer does not work,
  // and we have to use glReadPixels. Since GL_UNPACK_ROW_LENGTH is not
  // available in OpenGL ES 2, we should create the buffer so the pixels are
  // contiguous.
  //
  // TODO: verify if we can use kIOSurfaceBytesPerRow to force the
  // pool to give us contiguous data.
  return GetBufferWithoutPool(spec);
#else
  return pool->GetBuffer([this]() { FlushTextureCaches(); });
#endif  // TARGET_IPHONE_SIMULATOR
}

#else

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::MakeSimplePool(
    const BufferSpec& spec) {
  return GlTextureBufferPool::Create(spec.width, spec.height, spec.format,
                                     kKeepCount);
}

GpuBuffer GpuBufferMultiPool::GetBufferWithoutPool(const BufferSpec& spec) {
  return GpuBuffer(
      GlTextureBuffer::Create(spec.width, spec.height, spec.format));
}

GpuBuffer GpuBufferMultiPool::GetBufferFromSimplePool(
    BufferSpec spec, const GpuBufferMultiPool::SimplePool& pool) {
  return GpuBuffer(pool->GetBuffer());
}

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::RequestPool(
    const BufferSpec& spec) {
  SimplePool pool;
  std::vector<SimplePool> evicted;
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
  SimplePool pool = RequestPool(key);
  if (pool) {
    // Note: we release our multipool lock before accessing the simple pool.
    return GetBufferFromSimplePool(key, pool);
  } else {
    return GetBufferWithoutPool(key);
  }
}

GpuBufferMultiPool::~GpuBufferMultiPool() {
#ifdef __APPLE__
  CHECK_EQ(texture_caches_.size(), 0)
      << "Failed to unregister texture caches before deleting pool";
#endif  // defined(__APPLE__)
}

#ifdef __APPLE__
void GpuBufferMultiPool::RegisterTextureCache(CVTextureCacheType cache) {
  absl::MutexLock lock(&mutex_);

  CHECK(std::find(texture_caches_.begin(), texture_caches_.end(), cache) ==
        texture_caches_.end())
      << "Attempting to register a texture cache twice";
  texture_caches_.emplace_back(cache);
}

void GpuBufferMultiPool::UnregisterTextureCache(CVTextureCacheType cache) {
  absl::MutexLock lock(&mutex_);

  auto it = std::find(texture_caches_.begin(), texture_caches_.end(), cache);
  CHECK(it != texture_caches_.end())
      << "Attempting to unregister an unknown texture cache";
  texture_caches_.erase(it);
}
#endif  // defined(__APPLE__)

}  // namespace mediapipe
