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

#include "mediapipe/gpu/gl_calculator_helper_impl.h"

#if TARGET_OS_OSX
#import <AppKit/NSOpenGL.h>
#else
#import <OpenGLES/EAGL.h>
#endif  // TARGET_OS_OSX
#import <AVFoundation/AVFoundation.h>

#include "absl/memory/memory.h"
#include "mediapipe/gpu/gpu_buffer_multi_pool.h"
#include "mediapipe/gpu/pixel_buffer_pool_util.h"
#include "mediapipe/objc/util.h"

namespace mediapipe {

GlVersion GlCalculatorHelperImpl::GetGlVersion() {
#if TARGET_OS_OSX
  return GlVersion::kGL;
#else
  if (gl_context_->eagl_context().API == kEAGLRenderingAPIOpenGLES3) return GlVersion::kGLES3;
  else return GlVersion::kGLES2;
#endif  // TARGET_OS_OSX
}

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const mediapipe::ImageFrame& image_frame) {
  GlTexture texture;

  texture.helper_impl_ = this;
  texture.width_ = image_frame.Width();
  texture.height_ = image_frame.Height();
  auto format = GpuBufferFormatForImageFormat(image_frame.Format());

  GlTextureInfo info = GlTextureInfoForGpuBufferFormat(format, 0, GetGlVersion());

  glGenTextures(1, &texture.name_);
  glBindTexture(GL_TEXTURE_2D, texture.name_);
  glTexImage2D(GL_TEXTURE_2D, 0, info.gl_internal_format, texture.width_,
               texture.height_, 0, info.gl_format, info.gl_type,
               image_frame.PixelData());
  SetStandardTextureParams(GL_TEXTURE_2D, info.gl_internal_format);
  return texture;
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer) {
  return MapGpuBuffer(gpu_buffer, 0);
}

GlTexture GlCalculatorHelperImpl::CreateSourceTexture(
    const GpuBuffer& gpu_buffer, int plane) {
  return MapGpuBuffer(gpu_buffer, plane);
}

GlTexture GlCalculatorHelperImpl::MapGpuBuffer(
    const GpuBuffer& gpu_buffer, int plane) {
  CVReturn err;
  GlTexture texture;
  texture.helper_impl_ = this;
  texture.gpu_buffer_ = gpu_buffer;
  texture.plane_ = plane;

  const GlTextureInfo info =
      GlTextureInfoForGpuBufferFormat(gpu_buffer.format(), plane, GetGlVersion());
  // When scale is not 1, we still give the nominal size of the image.
  texture.width_ = gpu_buffer.width();
  texture.height_ = gpu_buffer.height();

#if TARGET_OS_OSX
  CVOpenGLTextureRef cv_texture_temp;
  err = CVOpenGLTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context_->cv_texture_cache(), gpu_buffer.GetCVPixelBufferRef(), NULL,
      &cv_texture_temp);
  NSCAssert(cv_texture_temp && !err,
            @"Error at CVOpenGLTextureCacheCreateTextureFromImage %d", err);
  texture.cv_texture_.adopt(cv_texture_temp);
  texture.target_ = CVOpenGLTextureGetTarget(*texture.cv_texture_);
  texture.name_ = CVOpenGLTextureGetName(*texture.cv_texture_);
#else
  CVOpenGLESTextureRef cv_texture_temp;
  err = CVOpenGLESTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context_->cv_texture_cache(), gpu_buffer.GetCVPixelBufferRef(), NULL,
      GL_TEXTURE_2D, info.gl_internal_format, texture.width_ / info.downscale,
      texture.height_ / info.downscale, info.gl_format, info.gl_type, plane,
      &cv_texture_temp);
  NSCAssert(cv_texture_temp && !err,
            @"Error at CVOpenGLESTextureCacheCreateTextureFromImage %d", err);
  texture.cv_texture_.adopt(cv_texture_temp);
  texture.target_ = CVOpenGLESTextureGetTarget(*texture.cv_texture_);
  texture.name_ = CVOpenGLESTextureGetName(*texture.cv_texture_);
#endif  // TARGET_OS_OSX

  glBindTexture(texture.target(), texture.name());
  SetStandardTextureParams(texture.target(), info.gl_internal_format);

  return texture;
}
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

template<>
std::unique_ptr<ImageFrame> GlTexture::GetFrame<ImageFrame>() const {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  if (gpu_buffer_.GetCVPixelBufferRef()) {
    return CreateImageFrameForCVPixelBuffer(gpu_buffer_.GetCVPixelBufferRef());
  }

  ImageFormat::Format image_format =
      ImageFormatForGpuBufferFormat(gpu_buffer_.format());
  // TODO: handle gl version here.
  GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      gpu_buffer_.format(), plane_);

  auto output = absl::make_unique<ImageFrame>(
      image_format, width_, height_);

  glReadPixels(0, 0, width_, height_, info.gl_format, info.gl_type,
               output->MutablePixelData());
  return output;
#else
  CHECK(gpu_buffer_.format() == GpuBufferFormat::kBGRA32);
  auto output =
      absl::make_unique<ImageFrame>(ImageFormat::SRGBA, width_, height_,
                                    ImageFrame::kGlDefaultAlignmentBoundary);

  CHECK(helper_impl_);
  helper_impl_->ReadTexture(*this, output->MutablePixelData(), output->PixelDataSize());

  return output;
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
}

template<>
std::unique_ptr<GpuBuffer> GlTexture::GetFrame<GpuBuffer>() const {
  NSCAssert(gpu_buffer_, @"gpu_buffer_ must be valid");
#if TARGET_IPHONE_SIMULATOR
  CVPixelBufferRef pixel_buffer = gpu_buffer_.GetCVPixelBufferRef();
  CVReturn err = CVPixelBufferLockBaseAddress(pixel_buffer, 0);
  NSCAssert(err == kCVReturnSuccess, @"CVPixelBufferLockBaseAddress failed: %d", err);
  OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
  uint8_t* pixel_ptr = static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixel_buffer));
  if (pixel_format == kCVPixelFormatType_32BGRA) {
    // TODO: restore previous framebuffer? Move this to helper so we can
    // use BindFramebuffer?
    glViewport(0, 0, width_, height_);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target_, name_, 0);

    size_t contiguous_bytes_per_row = width_ * 4;
    if (bytes_per_row == contiguous_bytes_per_row) {
      glReadPixels(0, 0, width_, height_, GL_BGRA, GL_UNSIGNED_BYTE, pixel_ptr);
    } else {
      std::vector<uint8_t> contiguous_buffer(contiguous_bytes_per_row * height_);
      uint8_t* temp_ptr = contiguous_buffer.data();
      glReadPixels(0, 0, width_, height_, GL_BGRA, GL_UNSIGNED_BYTE, temp_ptr);
      for (int i = 0; i < height_; ++i) {
        memcpy(pixel_ptr, temp_ptr, contiguous_bytes_per_row);
        temp_ptr += contiguous_bytes_per_row;
        pixel_ptr += bytes_per_row;
      }
    }
  } else {
    uint32_t format_big = CFSwapInt32HostToBig(pixel_format);
    NSLog(@"unsupported pixel format: %.4s", (char*)&format_big);
  }
  err = CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
  NSCAssert(err == kCVReturnSuccess, @"CVPixelBufferUnlockBaseAddress failed: %d", err);
#endif
  return absl::make_unique<GpuBuffer>(gpu_buffer_);
}

void GlTexture::Release() {
#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  if (*cv_texture_) {
    cv_texture_.reset(NULL);
  } else if (name_) {
    // This is only needed because of the glGenTextures in
    // CreateSourceTexture(ImageFrame)... change.
    glDeleteTextures(1, &name_);
  }
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
  helper_impl_ = nullptr;
  gpu_buffer_ = nullptr;
  plane_ = 0;
  name_ = 0;
  width_ = 0;
  height_ = 0;
}

}  // namespace mediapipe
