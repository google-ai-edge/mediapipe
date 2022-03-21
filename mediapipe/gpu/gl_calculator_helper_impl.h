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

#ifndef MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_IMPL_H_
#define MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_IMPL_H_

#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

#ifdef __OBJC__
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#endif  // __OBJC__

#ifdef __ANDROID__
#include "mediapipe/gpu/gl_texture_buffer_pool.h"
#endif

namespace mediapipe {

// This class implements the GlCalculatorHelper for iOS and Android.
// See GlCalculatorHelper for details on these methods.
class GlCalculatorHelperImpl {
 public:
  explicit GlCalculatorHelperImpl(CalculatorContext* cc,
                                  GpuResources* gpu_resources);
  ~GlCalculatorHelperImpl();

  absl::Status RunInGlContext(std::function<absl::Status(void)> gl_func,
                              CalculatorContext* calculator_context);

  GlTexture CreateSourceTexture(const ImageFrame& image_frame);
  GlTexture CreateSourceTexture(const GpuBuffer& gpu_buffer);

  // Note: multi-plane support is currently only available on iOS.
  GlTexture CreateSourceTexture(const GpuBuffer& gpu_buffer, int plane);

  // Creates a framebuffer and returns the texture that it is bound to.
  GlTexture CreateDestinationTexture(int output_width, int output_height,
                                     GpuBufferFormat format);

  GpuBuffer GpuBufferWithImageFrame(std::shared_ptr<ImageFrame> image_frame);
  GpuBuffer GpuBufferCopyingImageFrame(const ImageFrame& image_frame);

  GLuint framebuffer() const { return framebuffer_; }
  void BindFramebuffer(const GlTexture& dst);

  GlVersion GetGlVersion() const { return gl_context_->GetGlVersion(); }

  GlContext& GetGlContext() const;

  // For internal use.
  static void ReadTexture(const GlTextureView& view, void* output, size_t size);

 private:
  // Makes a GpuBuffer accessible as a texture in the GL context.
  GlTexture MapGpuBuffer(const GpuBuffer& gpu_buffer, GlTextureView view);

  // Create the framebuffer for rendering.
  void CreateFramebuffer();

  std::shared_ptr<GlContext> gl_context_;

  GLuint framebuffer_ = 0;

  GpuResources& gpu_resources_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GL_CALCULATOR_HELPER_IMPL_H_
