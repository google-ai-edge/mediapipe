// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/framework/formats/tensor.h"

#include <cstdint>
#include <utility>

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/logging.h"

#if MEDIAPIPE_METAL_ENABLED
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#else
#include <cstdlib>
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {

int BhwcBatchFromShape(const Tensor::Shape& shape) {
  LOG_IF(FATAL, shape.dims.empty())
      << "Tensor::Shape must be non-empty to retrieve a named dimension";
  return shape.dims[0];
}

int BhwcHeightFromShape(const Tensor::Shape& shape) {
  LOG_IF(FATAL, shape.dims.empty())
      << "Tensor::Shape must be non-empty to retrieve a named dimension";
  return shape.dims.size() < 4 ? 1 : shape.dims[shape.dims.size() - 3];
}

int BhwcWidthFromShape(const Tensor::Shape& shape) {
  LOG_IF(FATAL, shape.dims.empty())
      << "Tensor::Shape must be non-empty to retrieve a named dimension";
  return shape.dims.size() < 3 ? 1 : shape.dims[shape.dims.size() - 2];
}

int BhwcDepthFromShape(const Tensor::Shape& shape) {
  LOG_IF(FATAL, shape.dims.empty())
      << "Tensor::Shape must be non-empty to retrieve a named dimension";
  return shape.dims.size() < 2 ? 1 : shape.dims[shape.dims.size() - 1];
}

// TODO: Match channels count and padding for Texture2D:
// 1) support 1/2/4 channesl texture for 1/2/3-4 depth.
// 2) Allocate cpu_buffer_ with padded amount of memory
// 3) pad/"unpad" the bitmap after transfer CPU <-> GPU

#if MEDIAPIPE_METAL_ENABLED
namespace {
// MTLBuffer can use existing properly aligned and allocated CPU memory.
size_t AlignToPageSize(size_t size) {
  auto page_size = getpagesize();
  return (size + page_size - 1) / page_size * page_size;
}

void* AllocateVirtualMemory(size_t size) {
  vm_address_t data;
  auto error = vm_allocate(mach_task_self(), &data, AlignToPageSize(size),
                           VM_FLAGS_ANYWHERE);
  LOG_IF(FATAL, error != KERN_SUCCESS)
      << "Can't allocate virtual memory for Tensor.";
  return reinterpret_cast<void*>(data);
}

void DeallocateVirtualMemory(void* pointer, size_t size) {
  vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(pointer),
                size);
}
}  // namespace

Tensor::MtlBufferView Tensor::GetMtlBufferReadView(
    id<MTLCommandBuffer> command_buffer) const {
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  LOG_IF(FATAL, !(valid_ & (kValidCpu | kValidMetalBuffer)))
      << "Tensor conversion between different GPU resources is not supported "
         "yet.";
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  valid_ |= kValidMetalBuffer;
  AllocateMtlBuffer([command_buffer device]);
  return {metal_buffer_, std::move(lock)};
}

Tensor::MtlBufferView Tensor::GetMtlBufferWriteView(
    id<MTLCommandBuffer> command_buffer) const {
  // Don't overwrite command buffer at which the metal buffer has been written
  // so we can wait until completed.
  command_buffer_ = command_buffer;
  return GetMtlBufferWriteView([command_buffer device]);
}

Tensor::MtlBufferView Tensor::GetMtlBufferWriteView(
    id<MTLDevice> device) const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  valid_ = kValidMetalBuffer;
  AllocateMtlBuffer(device);
  return {metal_buffer_, std::move(lock)};
}

void Tensor::AllocateMtlBuffer(id<MTLDevice> device) const {
  device_ = device;
  if (!cpu_buffer_) {
    // It also means that the metal buffer is not allocated yet.
    cpu_buffer_ = AllocateVirtualMemory(bytes());
  }
  if (!metal_buffer_) {
    metal_buffer_ =
        [device_ newBufferWithBytesNoCopy:cpu_buffer_
                                   length:AlignToPageSize(bytes())
                                  options:MTLResourceStorageModeShared |
                                          MTLResourceCPUCacheModeDefaultCache
                              deallocator:^(void* pointer, NSUInteger length) {
                                DeallocateVirtualMemory(pointer, length);
                              }];
  }
}
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
Tensor::OpenGlTexture2dView Tensor::GetOpenGlTexture2dReadView() const {
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  LOG_IF(FATAL, !(valid_ & (kValidCpu | kValidOpenGlTexture2d)))
      << "Tensor conversion between different GPU resources is not supported "
         "yet.";
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  AllocateOpenGlTexture2d();
  if (!(valid_ & kValidOpenGlTexture2d)) {
    uint8_t* buffer;
    std::unique_ptr<uint8_t[]> temp_buffer;
    if (BhwcDepthFromShape(shape_) % 4 == 0) {
      // No padding exists because number of channels are multiple of 4.
      buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
    } else {
      const int padded_depth = (BhwcDepthFromShape(shape_) + 3) / 4 * 4;
      const int padded_depth_size = padded_depth * element_size();
      const int padded_size = BhwcBatchFromShape(shape_) *
                              BhwcHeightFromShape(shape_) *
                              BhwcWidthFromShape(shape_) * padded_depth_size;
      temp_buffer = absl::make_unique<uint8_t[]>(padded_size);
      buffer = temp_buffer.get();
      uint8_t* src_buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
      const int actual_depth_size = BhwcDepthFromShape(shape_) * element_size();
      for (int e = 0;
           e < BhwcBatchFromShape(shape_) * BhwcHeightFromShape(shape_) *
                   BhwcWidthFromShape(shape_);
           e++) {
        std::memcpy(buffer, src_buffer, actual_depth_size);
        src_buffer += actual_depth_size;
        buffer += padded_depth_size;
      }
    }
    // Transfer from CPU memory into GPU memory.
    glBindTexture(GL_TEXTURE_2D, opengl_texture2d_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, BhwcWidthFromShape(shape_),
                    BhwcHeightFromShape(shape_), GL_RGBA, GL_FLOAT, buffer);
    glBindTexture(GL_TEXTURE_2D, 0);
    valid_ |= kValidOpenGlTexture2d;
  }
  return {opengl_texture2d_, std::move(lock)};
}

Tensor::OpenGlTexture2dView Tensor::GetOpenGlTexture2dWriteView() const {
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  AllocateOpenGlTexture2d();
  valid_ = kValidOpenGlTexture2d;
  return {opengl_texture2d_, std::move(lock)};
}

void Tensor::AllocateOpenGlTexture2d() const {
  if (opengl_texture2d_ == GL_INVALID_INDEX) {
    gl_context_ = mediapipe::GlContext::GetCurrent();
    LOG_IF(FATAL, !gl_context_) << "GlContext is not bound to the thread.";
    glGenTextures(1, &opengl_texture2d_);
    glBindTexture(GL_TEXTURE_2D, opengl_texture2d_);
    // Texture2D represents a buffer with computable data so should be fetched
    // but not sampled - can affect performance. Also on GLES2.0 sampling is not
    // supported from floating point textures.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    const int pixels_per_depth = (BhwcDepthFromShape(shape_) + 3) / 4;
    const int width = BhwcWidthFromShape(shape_) * pixels_per_depth;
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, width,
                   BhwcHeightFromShape(shape_));
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenFramebuffers(1, &frame_buffer_);
  }
}
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
Tensor::OpenGlBufferView Tensor::GetOpenGlBufferReadView() const {
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  LOG_IF(FATAL, !(valid_ & (kValidCpu | kValidOpenGlBuffer)))
      << "Tensor conversion between different GPU resources is not supported "
         "yet.";
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  AllocateOpenGlBuffer();
  if (!(valid_ & kValidOpenGlBuffer)) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    void* ptr =
        glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes(),
                         GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_WRITE_BIT);
    std::memcpy(ptr, cpu_buffer_, bytes());
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    valid_ |= kValidOpenGlBuffer;
  }
  return {opengl_buffer_, std::move(lock)};
}

Tensor::OpenGlBufferView Tensor::GetOpenGlBufferWriteView() const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  AllocateOpenGlBuffer();
  valid_ = kValidOpenGlBuffer;
  return {opengl_buffer_, std::move(lock)};
}

void Tensor::AllocateOpenGlBuffer() const {
  if (opengl_buffer_ == GL_INVALID_INDEX) {
    gl_context_ = mediapipe::GlContext::GetCurrent();
    LOG_IF(FATAL, !gl_context_) << "GlContext is not bound to the thread.";
    glGenBuffers(1, &opengl_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes(), NULL, GL_STREAM_COPY);
  }
}
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

Tensor& Tensor::operator=(Tensor&& src) {
  if (this != &src) {
    Invalidate();
    Move(&src);
  }
  return *this;
}

void Tensor::Move(Tensor* src) {
  valid_ = src->valid_;
  src->valid_ = kValidNone;
  shape_ = src->shape();
  element_type_ = src->element_type();
  src->element_type_ = ElementType::kNone;  // Mark as invalidated.
  cpu_buffer_ = src->cpu_buffer_;
  src->cpu_buffer_ = nullptr;
#if MEDIAPIPE_METAL_ENABLED
  device_ = src->device_;
  command_buffer_ = src->command_buffer_;
  metal_buffer_ = src->metal_buffer_;
  src->metal_buffer_ = nil;
#endif  //  MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  gl_context_ = std::move(src->gl_context_);
  frame_buffer_ = src->frame_buffer_;
  src->frame_buffer_ = GL_INVALID_INDEX;
  opengl_texture2d_ = src->opengl_texture2d_;
  src->opengl_texture2d_ = GL_INVALID_INDEX;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  opengl_buffer_ = src->opengl_buffer_;
  src->opengl_buffer_ = GL_INVALID_INDEX;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
}

Tensor::Tensor(ElementType element_type, const Shape& shape)
    : element_type_(element_type), shape_(shape) {}

void Tensor::Invalidate() {
  absl::MutexLock lock(&view_mutex_);
#if MEDIAPIPE_METAL_ENABLED
  // If memory is allocated and not owned by the metal buffer.
  // TODO: Re-design cpu buffer memory management.
  if (cpu_buffer_ && !metal_buffer_) {
    DeallocateVirtualMemory(cpu_buffer_, AlignToPageSize(bytes()));
  }
  metal_buffer_ = nil;
#else
  if (cpu_buffer_) {
    free(cpu_buffer_);
  }
#endif  // MEDIAPIPE_METAL_ENABLED
  cpu_buffer_ = nullptr;

  // Don't need to wait for the resource to be deleted bacause if will be
  // released on last reference deletion inside the OpenGL driver.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  if (opengl_texture2d_ != GL_INVALID_INDEX) {
    GLuint opengl_texture2d = opengl_texture2d_;
    GLuint frame_buffer = frame_buffer_;
    gl_context_->RunWithoutWaiting([opengl_texture2d, frame_buffer]() {
      glDeleteTextures(1, &opengl_texture2d);
      glDeleteFramebuffers(1, &frame_buffer);
    });
    opengl_texture2d_ = GL_INVALID_INDEX;
    frame_buffer_ = GL_INVALID_INDEX;
  }
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  if (opengl_buffer_ != GL_INVALID_INDEX) {
    GLuint opengl_buffer = opengl_buffer_;
    gl_context_->RunWithoutWaiting(
        [opengl_buffer]() { glDeleteBuffers(1, &opengl_buffer); });
    opengl_buffer_ = GL_INVALID_INDEX;
  }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
}

Tensor::CpuReadView Tensor::GetCpuReadView() const {
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  AllocateCpuBuffer();
  if (!(valid_ & kValidCpu)) {
    // GPU-to-CPU synchronization and read-back.
#if MEDIAPIPE_METAL_ENABLED
    if (valid_ & kValidMetalBuffer) {
      LOG_IF(FATAL, !command_buffer_) << "Metal -> CPU synchronization "
                                         "requires MTLCommandBuffer to be set.";
      if (command_buffer_) {
        [command_buffer_ waitUntilCompleted];
      }
    }
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    if (valid_ & kValidOpenGlBuffer) {
      gl_context_->Run([this]() {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
        const void* ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes(),
                                           GL_MAP_READ_BIT);
        std::memcpy(cpu_buffer_, ptr, bytes());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
      });
    } else
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

        // Transfer data from texture if not transferred from SSBO/MTLBuffer
        // yet.
        if (valid_ & kValidOpenGlTexture2d) {
      gl_context_->Run([this]() {
        const int pixels_per_depth = (BhwcDepthFromShape(shape_) + 3) / 4;
        const int width = BhwcWidthFromShape(shape_) * pixels_per_depth;

        uint8_t* buffer;
        std::unique_ptr<uint8_t[]> temp_buffer;
        if (BhwcDepthFromShape(shape_) % 4 == 0) {
          buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
        } else {
          const int padded_size = BhwcBatchFromShape(shape_) *
                                  BhwcHeightFromShape(shape_) * width *
                                  pixels_per_depth * 4 * element_size();
          temp_buffer = absl::make_unique<uint8_t[]>(padded_size);
          buffer = temp_buffer.get();
        }

        glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, opengl_texture2d_, 0);
        glPixelStorei(GL_PACK_ROW_LENGTH, width);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, BhwcHeightFromShape(shape_), GL_RGBA,
                     GL_FLOAT, buffer);

        if (BhwcDepthFromShape(shape_) % 4) {
          uint8_t* dest_buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
          const int actual_depth_size =
              BhwcDepthFromShape(shape_) * element_size();
          const int padded_depth_size = pixels_per_depth * 4 * element_size();
          for (int e = 0;
               e < BhwcBatchFromShape(shape_) * BhwcHeightFromShape(shape_) *
                       BhwcWidthFromShape(shape_);
               e++) {
            std::memcpy(dest_buffer, buffer, actual_depth_size);
            dest_buffer += actual_depth_size;
            buffer += padded_depth_size;
          }
        }
      });
    }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
    valid_ |= kValidCpu;
  }
  return {cpu_buffer_, std::move(lock)};
}

Tensor::CpuWriteView Tensor::GetCpuWriteView() const {
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  AllocateCpuBuffer();
  valid_ = kValidCpu;
  return {cpu_buffer_, std::move(lock)};
}

void Tensor::AllocateCpuBuffer() const {
  if (!cpu_buffer_) {
#if MEDIAPIPE_METAL_ENABLED
    cpu_buffer_ = AllocateVirtualMemory(bytes());
#else
    cpu_buffer_ = malloc(bytes());
#endif  // MEDIAPIPE_METAL_ENABLED
  }
}

}  // namespace mediapipe
