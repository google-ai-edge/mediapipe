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

#ifdef __APPLE__
#include "mediapipe/objc/CFHolder.h"
#endif  // __APPLE__

namespace mediapipe {

// Keep this many buffers allocated for a given frame size.
static constexpr int kKeepCount = 2;
// The maximum size of the GpuBufferMultiPool. When the limit is reached, the
// oldest BufferSpec will be dropped.
static constexpr int kMaxPoolCount = 20;

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::MakeSimplePool(
    const BufferSpec& spec) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(spec.format);
  CHECK_NE(cv_format, -1) << "unsupported pixel format";
  return MakeCFHolderAdopting(
      CreateCVPixelBufferPool(spec.width, spec.height, cv_format, kKeepCount,
                              0.1 /* max age in seconds */));
}

GpuBuffer GpuBufferMultiPool::GetBufferFromSimplePool(
    BufferSpec spec, const GpuBufferMultiPool::SimplePool& pool) {
#if TARGET_IPHONE_SIMULATOR
  // On the simulator, syncing the texture with the pixelbuffer does not work,
  // and we have to use glReadPixels. Since GL_UNPACK_ROW_LENGTH is not
  // available in OpenGL ES 2, we should create the buffer so the pixels are
  // contiguous.
  //
  // TODO: verify if we can use kIOSurfaceBytesPerRow to force the
  // pool to give us contiguous data.
  OSType cv_format = CVPixelFormatForGpuBufferFormat(spec.format);
  CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err = CreateCVPixelBufferWithoutPool(spec.width, spec.height,
                                                cv_format, &buffer);
  CHECK(!err) << "Error creating pixel buffer: " << err;
  return GpuBuffer(MakeCFHolderAdopting(buffer));
#else
  CVPixelBufferRef buffer;
  // TODO: allow the keepCount and the allocation threshold to be set
  // by the application, and to be set independently.
  static CFDictionaryRef auxAttributes =
      CreateCVPixelBufferPoolAuxiliaryAttributesForThreshold(kKeepCount);
  CVReturn err = CreateCVPixelBufferWithPool(
      *pool, auxAttributes,
      [this]() {
        for (const auto& cache : texture_caches_) {
#if TARGET_OS_OSX
          CVOpenGLTextureCacheFlush(*cache, 0);
#else
          CVOpenGLESTextureCacheFlush(*cache, 0);
#endif  // TARGET_OS_OSX
        }
      },
      &buffer);
  CHECK(!err) << "Error creating pixel buffer: " << err;
  return GpuBuffer(MakeCFHolderAdopting(buffer));
#endif  // TARGET_IPHONE_SIMULATOR
}

#else

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::MakeSimplePool(
    const BufferSpec& spec) {
  return GlTextureBufferPool::Create(spec.width, spec.height, spec.format,
                                     kKeepCount);
}

GpuBuffer GpuBufferMultiPool::GetBufferFromSimplePool(
    BufferSpec spec, const GpuBufferMultiPool::SimplePool& pool) {
  return GpuBuffer(pool->GetBuffer());
}

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

GpuBufferMultiPool::SimplePool GpuBufferMultiPool::GetSimplePool(
    const BufferSpec& key) {
  absl::MutexLock lock(&mutex_);
  auto pool_it = pools_.find(key);
  if (pool_it == pools_.end()) {
    // Discard the least recently used pool in LRU cache.
    if (pools_.size() >= kMaxPoolCount) {
      auto old_spec = buffer_specs_.front();  // Front has LRU.
      buffer_specs_.pop_front();
      pools_.erase(old_spec);
    }
    buffer_specs_.push_back(key);  // Push new spec to back.
    std::tie(pool_it, std::ignore) =
        pools_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                       std::forward_as_tuple(MakeSimplePool(key)));
  } else {
    // Find and move current 'key' spec to back, keeping others in same order.
    auto specs_it = buffer_specs_.begin();
    while (specs_it != buffer_specs_.end()) {
      if (*specs_it == key) {
        buffer_specs_.erase(specs_it);
        break;
      }
      ++specs_it;
    }
    buffer_specs_.push_back(key);
  }
  return pool_it->second;
}

GpuBuffer GpuBufferMultiPool::GetBuffer(int width, int height,
                                        GpuBufferFormat format) {
  BufferSpec key(width, height, format);
  SimplePool pool = GetSimplePool(key);
  // Note: we release our multipool lock before accessing the simple pool.
  return GetBufferFromSimplePool(key, pool);
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
