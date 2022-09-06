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
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include "mediapipe/gpu/gl_base.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#if MEDIAPIPE_METAL_ENABLED
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#else
#include <cstdlib>
#endif  // MEDIAPIPE_METAL_ENABLED

namespace mediapipe {

// Zero and negative values are not checked here.
bool IsPowerOfTwo(int v) { return (v & (v - 1)) == 0; }

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
bool Tensor::NeedsHalfFloatRenderTarget() const {
  static bool has_color_buffer_float =
      gl_context_->HasGlExtension("WEBGL_color_buffer_float") ||
      gl_context_->HasGlExtension("EXT_color_buffer_float");
  if (!has_color_buffer_float) {
    static bool has_color_buffer_half_float =
        gl_context_->HasGlExtension("EXT_color_buffer_half_float");
    LOG_IF(FATAL, !has_color_buffer_half_float)
        << "EXT_color_buffer_half_float or WEBGL_color_buffer_float "
        << "required on web to use MP tensor";
    return true;
  }
  return false;
}

Tensor::OpenGlTexture2dView Tensor::GetOpenGlTexture2dReadView() const {
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  LOG_IF(FATAL, !(valid_ & (kValidCpu | kValidOpenGlTexture2d)))
      << "Tensor conversion between different GPU resources is not supported "
         "yet.";
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  AllocateOpenGlTexture2d();
  if (!(valid_ & kValidOpenGlTexture2d)) {
    const int padded_size =
        texture_height_ * texture_width_ * 4 * element_size();
    auto temp_buffer = absl::make_unique<uint8_t[]>(padded_size);
    uint8_t* dest_buffer = temp_buffer.get();
    uint8_t* src_buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
    const int num_elements = BhwcWidthFromShape(shape_) *
                             BhwcHeightFromShape(shape_) *
                             BhwcBatchFromShape(shape_);
    const int actual_depth_size = BhwcDepthFromShape(shape_) * element_size();
    const int padded_depth_size =
        (BhwcDepthFromShape(shape_) + 3) / 4 * 4 * element_size();
    for (int e = 0; e < num_elements; e++) {
      std::memcpy(dest_buffer, src_buffer, actual_depth_size);
      src_buffer += actual_depth_size;
      dest_buffer += padded_depth_size;
    }
    // Transfer from CPU memory into GPU memory.
    glBindTexture(GL_TEXTURE_2D, opengl_texture2d_);
    // Set alignment for the proper value (default) to avoid address sanitizer
    // error "out of boundary reading".
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
#ifdef __EMSCRIPTEN__
    // Under WebGL1, format must match in order to use glTexSubImage2D, so if we
    // have a half-float texture, then uploading from GL_FLOAT here would fail.
    // We change the texture's data type to float here to accommodate.
    // Furthermore, for a full-image replacement operation, glTexImage2D is
    // expected to be more performant than glTexSubImage2D. Note that for WebGL2
    // we cannot use glTexImage2D, because we allocate using glTexStorage2D in
    // that case, which is incompatible.
    if (gl_context_->GetGlVersion() == mediapipe::GlVersion::kGLES2) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width_, texture_height_,
                   0, GL_RGBA, GL_FLOAT, temp_buffer.get());
      texture_is_half_float_ = false;
    } else
#endif  // __EMSCRIPTEN__
    {
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texture_width_, texture_height_,
                      GL_RGBA, GL_FLOAT, temp_buffer.get());
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    valid_ |= kValidOpenGlTexture2d;
  }
  return {opengl_texture2d_, std::move(lock)};
}

Tensor::OpenGlTexture2dView Tensor::GetOpenGlTexture2dWriteView() const {
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  AllocateOpenGlTexture2d();
#ifdef __EMSCRIPTEN__
  // On web, we may have to change type from float to half-float
  if (!texture_is_half_float_ && NeedsHalfFloatRenderTarget()) {
    glBindTexture(GL_TEXTURE_2D, opengl_texture2d_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width_, texture_height_, 0,
                 GL_RGBA, GL_HALF_FLOAT_OES, 0 /* data */);
    glBindTexture(GL_TEXTURE_2D, 0);
    texture_is_half_float_ = true;
  }
#endif
  valid_ = kValidOpenGlTexture2d;
  return {opengl_texture2d_, std::move(lock)};
}

Tensor::OpenGlTexture2dView::Layout
Tensor::OpenGlTexture2dView::GetLayoutDimensions(const Tensor::Shape& shape,
                                                 int* width, int* height) {
  static int max_size = 0;
  if (max_size == 0) {
    int max_texture_size;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_texture_size);
    int max_renderbuffer_size;
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &max_renderbuffer_size);
    int max_viewport_dims[2];
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_viewport_dims);
    max_size = std::min(std::min(max_texture_size, max_renderbuffer_size),
                        std::min(max_viewport_dims[0], max_viewport_dims[1]));
  }
  const int num_slices = (BhwcDepthFromShape(shape) + 3) / 4;
  const int num_elements = BhwcBatchFromShape(shape) *
                           BhwcHeightFromShape(shape) *
                           BhwcWidthFromShape(shape);
  const int num_pixels = num_slices * num_elements;
  int w = BhwcWidthFromShape(shape) * num_slices;
  if (w <= max_size) {
    int h = (num_pixels + w - 1) / w;
    if (h <= max_size) {
      *width = w;
      *height = h;
      return Tensor::OpenGlTexture2dView::Layout::kAligned;
    }
  }
  // The best performance of a compute shader can be achived with textures'
  // width multiple of 256. Making minimum fixed width of 256 waste memory for
  // small tensors. The optimal balance memory-vs-performance is power of 2.
  // The texture width and height are choosen to be closer to square.
  float power = std::log2(std::sqrt(static_cast<float>(num_pixels)));
  w = 1 << static_cast<int>(power);
  int h = (num_pixels + w - 1) / w;
  LOG_IF(FATAL, w > max_size || h > max_size)
      << "The tensor can't fit into OpenGL Texture2D View.";
  *width = w;
  *height = h;
  return Tensor::OpenGlTexture2dView::Layout::kLinearized;
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
    OpenGlTexture2dView::GetLayoutDimensions(shape_, &texture_width_,
                                             &texture_height_);
    if (gl_context_->GetGlVersion() != mediapipe::GlVersion::kGLES2) {
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, texture_width_,
                     texture_height_);
    } else {
      // GLES2.0 supports only clamp addressing mode for NPOT textures.
      // If any of dimensions is NPOT then both addressing modes are clamp.
      if (!IsPowerOfTwo(texture_width_) || !IsPowerOfTwo(texture_height_)) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      }
      // We assume all contexts will have the same extensions, so we only check
      // once for OES_texture_float extension, to save time.
      static bool has_oes_extension =
          gl_context_->HasGlExtension("OES_texture_float");
      LOG_IF(FATAL, !has_oes_extension)
          << "OES_texture_float extension required in order to use MP tensor "
          << "with GLES 2.0";
      // Allocate the image data; note that it's no longer RGBA32F, so will be
      // lower precision.
      auto type = GL_FLOAT;
      // On web, we might need to change type to half-float (e.g. for iOS-
      // Safari) in order to have a valid framebuffer. See b/194442743 for more
      // details.
#ifdef __EMSCRIPTEN__
      if (NeedsHalfFloatRenderTarget()) {
        type = GL_HALF_FLOAT_OES;
        texture_is_half_float_ = true;
      }
#endif  // __EMSCRIPTEN
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_width_, texture_height_,
                   0, GL_RGBA, type, 0 /* data */);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenFramebuffers(1, &frame_buffer_);
  }
}
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
Tensor::OpenGlBufferView Tensor::GetOpenGlBufferReadView() const {
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
  LOG_IF(FATAL, !(valid_ & (kValidCpu |
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
                            kValidAHardwareBuffer |
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
                            kValidOpenGlBuffer)))
      << "Tensor conversion between different GPU resources is not supported.";
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  AllocateOpenGlBuffer();
  if (!(valid_ & kValidOpenGlBuffer)) {
    // If the call succeds then AHWB -> SSBO are synchronized so any usage of
    // the SSBO is correct after this call.
    if (!InsertAhwbToSsboFence()) {
      glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
      void* ptr =
          glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, bytes(),
                           GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_WRITE_BIT);
      CHECK(ptr) << "glMapBufferRange failed: " << glGetError();
      std::memcpy(ptr, cpu_buffer_, bytes());
      glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    valid_ |= kValidOpenGlBuffer;
  }
  return {opengl_buffer_, std::move(lock),
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
          &ssbo_read_
#else
          nullptr
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
  };
}

Tensor::OpenGlBufferView Tensor::GetOpenGlBufferWriteView() const {
  auto lock(absl::make_unique<absl::MutexLock>(&view_mutex_));
  AllocateOpenGlBuffer();
  valid_ = kValidOpenGlBuffer;
  return {opengl_buffer_, std::move(lock), nullptr};
}

void Tensor::AllocateOpenGlBuffer() const {
  if (opengl_buffer_ == GL_INVALID_INDEX) {
    gl_context_ = mediapipe::GlContext::GetCurrent();
    LOG_IF(FATAL, !gl_context_) << "GlContext is not bound to the thread.";
    glGenBuffers(1, &opengl_buffer_);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, opengl_buffer_);
    if (!AllocateAhwbMapToSsbo()) {
      glBufferData(GL_SHADER_STORAGE_BUFFER, bytes(), NULL, GL_STREAM_COPY);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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

  MoveAhwbStuff(src);

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  gl_context_ = std::move(src->gl_context_);
  frame_buffer_ = src->frame_buffer_;
  src->frame_buffer_ = GL_INVALID_INDEX;
  opengl_texture2d_ = src->opengl_texture2d_;
  src->opengl_texture2d_ = GL_INVALID_INDEX;
  texture_width_ = src->texture_width_;
  texture_height_ = src->texture_height_;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  opengl_buffer_ = src->opengl_buffer_;
  src->opengl_buffer_ = GL_INVALID_INDEX;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
}

Tensor::Tensor(ElementType element_type, const Shape& shape)
    : element_type_(element_type), shape_(shape) {}
Tensor::Tensor(ElementType element_type, const Shape& shape,
               const QuantizationParameters& quantization_parameters)
    : element_type_(element_type),
      shape_(shape),
      quantization_parameters_(quantization_parameters) {}

#if MEDIAPIPE_METAL_ENABLED
void Tensor::Invalidate() {
  absl::MutexLock lock(&view_mutex_);
  // If memory is allocated and not owned by the metal buffer.
  // TODO: Re-design cpu buffer memory management.
  if (cpu_buffer_ && !metal_buffer_) {
    DeallocateVirtualMemory(cpu_buffer_, AlignToPageSize(bytes()));
  }
  metal_buffer_ = nil;
  cpu_buffer_ = nullptr;
}

#else

void Tensor::Invalidate() {
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  GLuint cleanup_gl_tex = GL_INVALID_INDEX;
  GLuint cleanup_gl_fb = GL_INVALID_INDEX;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  GLuint cleanup_gl_buf = GL_INVALID_INDEX;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  {
    absl::MutexLock lock(&view_mutex_);
    ReleaseAhwbStuff();

    // Don't need to wait for the resource to be deleted bacause if will be
    // released on last reference deletion inside the OpenGL driver.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
    std::swap(cleanup_gl_tex, opengl_texture2d_);
    std::swap(cleanup_gl_fb, frame_buffer_);
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
    std::swap(cleanup_gl_buf, opengl_buffer_);
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  }
  // Do not hold the view mutex while invoking GlContext::RunWithoutWaiting,
  // since that method may acquire the context's own lock.
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  if (cleanup_gl_tex != GL_INVALID_INDEX || cleanup_gl_fb != GL_INVALID_INDEX ||
      cleanup_gl_buf != GL_INVALID_INDEX) {
    gl_context_->RunWithoutWaiting(
        [cleanup_gl_tex, cleanup_gl_fb, cleanup_gl_buf]() {
          glDeleteTextures(1, &cleanup_gl_tex);
          glDeleteFramebuffers(1, &cleanup_gl_fb);
          glDeleteBuffers(1, &cleanup_gl_buf);
        });
  }
#elif MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  if (cleanup_gl_tex != GL_INVALID_INDEX || cleanup_gl_fb != GL_INVALID_INDEX) {
    gl_context_->RunWithoutWaiting([cleanup_gl_tex, cleanup_gl_fb]() {
      glDeleteTextures(1, &cleanup_gl_tex);
      glDeleteFramebuffers(1, &cleanup_gl_fb);
    });
  }
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

  if (cpu_buffer_) {
    free(cpu_buffer_);
  }
  cpu_buffer_ = nullptr;
}
#endif  // MEDIAPIPE_METAL_ENABLED

Tensor::CpuReadView Tensor::GetCpuReadView() const {
  auto lock = absl::make_unique<absl::MutexLock>(&view_mutex_);
  LOG_IF(FATAL, valid_ == kValidNone)
      << "Tensor must be written prior to read from.";
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  if (__builtin_available(android 26, *)) {
    void* ptr = MapAhwbToCpuRead();
    if (ptr) {
      valid_ |= kValidCpu;
      return {ptr, std::move(lock), [ahwb = ahwb_] {
                auto error = AHardwareBuffer_unlock(ahwb, nullptr);
                CHECK(error == 0) << "AHardwareBuffer_unlock " << error;
              }};
    }
  }
#endif  // MEDIAPIPE_TENSOR_USE_AHWB

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
    // TODO: we cannot just grab the GL context's lock while holding
    // the view mutex here.
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
          const int padded_size =
              texture_height_ * texture_width_ * 4 * element_size();
          auto temp_buffer = absl::make_unique<uint8_t[]>(padded_size);
          uint8_t* buffer = temp_buffer.get();

          glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_);
          glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                 GL_TEXTURE_2D, opengl_texture2d_, 0);
          glPixelStorei(GL_PACK_ALIGNMENT, 4);
          glReadPixels(0, 0, texture_width_, texture_height_, GL_RGBA, GL_FLOAT,
                       buffer);
          uint8_t* dest_buffer = reinterpret_cast<uint8_t*>(cpu_buffer_);
          const int actual_depth_size =
              BhwcDepthFromShape(shape_) * element_size();
          const int num_slices = (BhwcDepthFromShape(shape_) + 3) / 4;
          const int padded_depth_size = num_slices * 4 * element_size();
          const int num_elements = BhwcWidthFromShape(shape_) *
                                   BhwcHeightFromShape(shape_) *
                                   BhwcBatchFromShape(shape_);
          for (int e = 0; e < num_elements; e++) {
            std::memcpy(dest_buffer, buffer, actual_depth_size);
            dest_buffer += actual_depth_size;
            buffer += padded_depth_size;
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
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  if (__builtin_available(android 26, *)) {
    void* ptr = MapAhwbToCpuWrite();
    if (ptr) {
      return {ptr, std::move(lock), [ahwb = ahwb_, fence_fd = &fence_fd_] {
                auto error = AHardwareBuffer_unlock(ahwb, fence_fd);
                CHECK(error == 0) << "AHardwareBuffer_unlock " << error;
              }};
    }
  }
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
  return {cpu_buffer_, std::move(lock)};
}

void Tensor::AllocateCpuBuffer() const {
  if (!cpu_buffer_) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
    if (AllocateAHardwareBuffer()) return;
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
#if MEDIAPIPE_METAL_ENABLED
    cpu_buffer_ = AllocateVirtualMemory(bytes());
#else
    cpu_buffer_ = malloc(bytes());
#endif  // MEDIAPIPE_METAL_ENABLED
  }
}

void Tensor::SetPreferredStorageType(StorageType type) {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  if (__builtin_available(android 26, *)) {
    use_ahwb_ = type == StorageType::kAhwb;
    VLOG(4) << "Tensor: use of AHardwareBuffer is "
            << (use_ahwb_ ? "allowed" : "not allowed");
  }
#else
  VLOG(4) << "Tensor: use of AHardwareBuffer is not allowed";
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
}

Tensor::StorageType Tensor::GetPreferredStorageType() {
#ifdef MEDIAPIPE_TENSOR_USE_AHWB
  return use_ahwb_ ? StorageType::kAhwb : StorageType::kDefault;
#else
  return StorageType::kDefault;
#endif  // MEDIAPIPE_TENSOR_USE_AHWB
}

}  // namespace mediapipe
