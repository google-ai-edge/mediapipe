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

#include "mediapipe/framework/formats/image_multi_pool.h"

#include <tuple>

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"

#if !MEDIAPIPE_DISABLE_GPU
#ifdef __APPLE__
#include "mediapipe/objc/CFHolder.h"
#include "mediapipe/objc/util.h"
#endif  // __APPLE__
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace mediapipe {

// Keep this many buffers allocated for a given frame size.
static constexpr int kKeepCount = 2;
// The maximum size of the ImageMultiPool. When the limit is reached, the
// oldest IBufferSpec will be dropped.
static constexpr int kMaxPoolCount = 20;

#if !MEDIAPIPE_DISABLE_GPU

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

ImageMultiPool::SimplePoolGpu ImageMultiPool::MakeSimplePoolGpu(
    IBufferSpec spec) {
  OSType cv_format = mediapipe::CVPixelFormatForGpuBufferFormat(
      GpuBufferFormatForImageFormat(spec.format));
  ABSL_CHECK_NE(cv_format, -1) << "unsupported pixel format";
  return MakeCFHolderAdopting(mediapipe::CreateCVPixelBufferPool(
      spec.width, spec.height, cv_format, kKeepCount,
      0.1 /* max age in seconds */));
}

Image ImageMultiPool::GetBufferFromSimplePool(
    IBufferSpec spec, const ImageMultiPool::SimplePoolGpu& pool) {
#if TARGET_IPHONE_SIMULATOR
  // On the simulator, syncing the texture with the pixelbuffer does not work,
  // and we have to use glReadPixels. Since GL_UNPACK_ROW_LENGTH is not
  // available in OpenGL ES 2, we should create the buffer so the pixels are
  // contiguous.
  //
  // TODO: verify if we can use kIOSurfaceBytesPerRow to force the
  // pool to give us contiguous data.
  OSType cv_format = mediapipe::CVPixelFormatForGpuBufferFormat(
      mediapipe::GpuBufferFormatForImageFormat(spec.format));
  ABSL_CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err = mediapipe::CreateCVPixelBufferWithoutPool(
      spec.width, spec.height, cv_format, &buffer);
  ABSL_CHECK(!err) << "Error creating pixel buffer: " << err;
  return Image(MakeCFHolderAdopting(buffer));
#else
  CVPixelBufferRef buffer;
  // TODO: allow the keepCount and the allocation threshold to be set
  // by the application, and to be set independently.
  static CFDictionaryRef auxAttributes =
      mediapipe::CreateCVPixelBufferPoolAuxiliaryAttributesForThreshold(
          kKeepCount);
  CVReturn err = mediapipe::CreateCVPixelBufferWithPool(
      *pool, auxAttributes,
      [this]() {
        absl::MutexLock lock(&mutex_gpu_);
        for (const auto& cache : texture_caches_) {
#if TARGET_OS_OSX
          CVOpenGLTextureCacheFlush(*cache, 0);
#else
          CVOpenGLESTextureCacheFlush(*cache, 0);
#endif  // TARGET_OS_OSX
        }
      },
      &buffer);
  ABSL_CHECK(!err) << "Error creating pixel buffer: " << err;
  return Image(MakeCFHolderAdopting(buffer));
#endif  // TARGET_IPHONE_SIMULATOR
}

#else

ImageMultiPool::SimplePoolGpu ImageMultiPool::MakeSimplePoolGpu(
    IBufferSpec spec) {
  return mediapipe::GlTextureBufferPool::Create(
      spec.width, spec.height, GpuBufferFormatForImageFormat(spec.format),
      kKeepCount);
}

Image ImageMultiPool::GetBufferFromSimplePool(
    IBufferSpec spec, const ImageMultiPool::SimplePoolGpu& pool) {
  auto buffer = pool->GetBuffer();
  ABSL_CHECK_OK(buffer);
  return Image(*std::move(buffer));
}

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

#endif  // !MEDIAPIPE_DISABLE_GPU

ImageMultiPool::SimplePoolCpu ImageMultiPool::MakeSimplePoolCpu(
    IBufferSpec spec) {
  return ImageFramePool::Create(spec.width, spec.height, spec.format,
                                kKeepCount);
}

Image ImageMultiPool::GetBufferFromSimplePool(
    IBufferSpec spec, const ImageMultiPool::SimplePoolCpu& pool) {
  return Image(pool->GetBuffer());
}

Image ImageMultiPool::GetBuffer(int width, int height, bool use_gpu,
                                ImageFormat::Format format) {
#if !MEDIAPIPE_DISABLE_GPU
  if (use_gpu) {
    absl::MutexLock lock(&mutex_gpu_);
    IBufferSpec key(width, height, format);
    auto pool_it = pools_gpu_.find(key);
    if (pool_it == pools_gpu_.end()) {
      // Discard the least recently used pool in LRU cache.
      if (pools_gpu_.size() >= kMaxPoolCount) {
        auto old_spec = buffer_specs_gpu_.front();  // Front has LRU.
        buffer_specs_gpu_.pop_front();
        pools_gpu_.erase(old_spec);
      }
      buffer_specs_gpu_.push_back(key);  // Push new spec to back.
      std::tie(pool_it, std::ignore) = pools_gpu_.emplace(
          std::piecewise_construct, std::forward_as_tuple(key),
          std::forward_as_tuple(MakeSimplePoolGpu(key)));
    } else {
      // Find and move current 'key' spec to back, keeping others in same order.
      auto specs_it = buffer_specs_gpu_.begin();
      while (specs_it != buffer_specs_gpu_.end()) {
        if (*specs_it == key) {
          buffer_specs_gpu_.erase(specs_it);
          break;
        }
        ++specs_it;
      }
      buffer_specs_gpu_.push_back(key);
    }
    return GetBufferFromSimplePool(pool_it->first, pool_it->second);
  } else  // NOLINT(readability/braces)
#endif    // !MEDIAPIPE_DISABLE_GPU
  {
    absl::MutexLock lock(&mutex_cpu_);
    IBufferSpec key(width, height, format);
    auto pool_it = pools_cpu_.find(key);
    if (pool_it == pools_cpu_.end()) {
      // Discard the least recently used pool in LRU cache.
      if (pools_cpu_.size() >= kMaxPoolCount) {
        auto old_spec = buffer_specs_cpu_.front();  // Front has LRU.
        buffer_specs_cpu_.pop_front();
        pools_cpu_.erase(old_spec);
      }
      buffer_specs_cpu_.push_back(key);  // Push new spec to back.
      std::tie(pool_it, std::ignore) = pools_cpu_.emplace(
          std::piecewise_construct, std::forward_as_tuple(key),
          std::forward_as_tuple(MakeSimplePoolCpu(key)));
    } else {
      // Find and move current 'key' spec to back, keeping others in same order.
      auto specs_it = buffer_specs_cpu_.begin();
      while (specs_it != buffer_specs_cpu_.end()) {
        if (*specs_it == key) {
          buffer_specs_cpu_.erase(specs_it);
          break;
        }
        ++specs_it;
      }
      buffer_specs_cpu_.push_back(key);
    }
    return GetBufferFromSimplePool(pool_it->first, pool_it->second);
  }
}

ImageMultiPool::~ImageMultiPool() {
#if !MEDIAPIPE_DISABLE_GPU
#ifdef __APPLE__
  ABSL_CHECK_EQ(texture_caches_.size(), 0)
      << "Failed to unregister texture caches before deleting pool";
#endif  // defined(__APPLE__)
#endif  // !MEDIAPIPE_DISABLE_GPU
}

#if !MEDIAPIPE_DISABLE_GPU
#ifdef __APPLE__
void ImageMultiPool::RegisterTextureCache(mediapipe::CVTextureCacheType cache) {
  absl::MutexLock lock(&mutex_gpu_);

  ABSL_CHECK(std::find(texture_caches_.begin(), texture_caches_.end(), cache) ==
             texture_caches_.end())
      << "Attempting to register a texture cache twice";
  texture_caches_.emplace_back(cache);
}

void ImageMultiPool::UnregisterTextureCache(
    mediapipe::CVTextureCacheType cache) {
  absl::MutexLock lock(&mutex_gpu_);

  auto it = std::find(texture_caches_.begin(), texture_caches_.end(), cache);
  ABSL_CHECK(it != texture_caches_.end())
      << "Attempting to unregister an unknown texture cache";
  texture_caches_.erase(it);
}
#endif  // defined(__APPLE__)
#endif  // !MEDIAPIPE_DISABLE_GPU

}  // namespace mediapipe
