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

#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/egl_surface_holder.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_surface_sink_calculator.pb.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {
namespace api2 {

enum { kAttribVertex, kAttribTexturePosition, kNumberOfAttributes };

// Receives GpuBuffers and renders them to an EGL surface.
// Can be used to render to an Android SurfaceTexture.
//
// Inputs:
//   VIDEO or index 0: GpuBuffers to be rendered.
// Side inputs:
//   SURFACE: unique_ptr to an EglSurfaceHolder to draw to.
//   GPU_SHARED: shared GPU resources.
//
// See GlSurfaceSinkCalculatorOptions for options.
class GlSurfaceSinkCalculator : public Node {
 public:
  static constexpr Input<
      OneOf<mediapipe::Image, mediapipe::GpuBuffer>>::Optional kInVideo{
      "VIDEO"};
  static constexpr Input<
      OneOf<mediapipe::Image, mediapipe::GpuBuffer>>::Optional kIn{""};
  static constexpr SideInput<std::unique_ptr<mediapipe::EglSurfaceHolder>>
      kSurface{"SURFACE"};

  MEDIAPIPE_NODE_INTERFACE(GlSurfaceSinkCalculator, kInVideo, kIn, kSurface);

  ~GlSurfaceSinkCalculator();

  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) final;
  absl::Status Process(CalculatorContext* cc) final;

 private:
  mediapipe::GlCalculatorHelper helper_;
  mediapipe::EglSurfaceHolder* surface_holder_;
  bool initialized_ = false;
  std::unique_ptr<mediapipe::QuadRenderer> renderer_;
  mediapipe::FrameScaleMode scale_mode_ =
      mediapipe::FrameScaleMode::kFillAndCrop;
};
MEDIAPIPE_REGISTER_NODE(GlSurfaceSinkCalculator);

// static
absl::Status GlSurfaceSinkCalculator::UpdateContract(CalculatorContract* cc) {
  RET_CHECK(kInVideo(cc).IsConnected() ^ kIn(cc).IsConnected())
      << "Only one of VIDEO or index 0 input is expected.";

  // Currently we pass GL context information and other stuff as external
  // inputs, which are handled by the helper.
  return mediapipe::GlCalculatorHelper::UpdateContract(cc);
}

absl::Status GlSurfaceSinkCalculator::Open(CalculatorContext* cc) {
  surface_holder_ = kSurface(cc).Get().get();

  scale_mode_ = FrameScaleModeFromProto(
      cc->Options<mediapipe::GlSurfaceSinkCalculatorOptions>()
          .frame_scale_mode(),
      mediapipe::FrameScaleMode::kFillAndCrop);

  // Let the helper access the GL context information.
  return helper_.Open(cc);
}

absl::Status GlSurfaceSinkCalculator::Process(CalculatorContext* cc) {
  return helper_.RunInGlContext([this, &cc]() -> absl::Status {
    absl::MutexLock lock(&surface_holder_->mutex);
    EGLSurface surface = surface_holder_->surface;
    if (surface == EGL_NO_SURFACE) {
      LOG_EVERY_N(INFO, 300) << "GlSurfaceSinkCalculator: no surface";
      return absl::OkStatus();
    }

    mediapipe::Packet packet;
    if (kInVideo(cc).IsConnected())
      packet = kInVideo(cc).packet();
    else
      packet = kIn(cc).packet();

    mediapipe::GpuBuffer input;
    if (packet.ValidateAsType<mediapipe::GpuBuffer>().ok())
      input = packet.Get<mediapipe::GpuBuffer>();
    if (packet.ValidateAsType<mediapipe::Image>().ok())
      input = packet.Get<mediapipe::Image>().GetGpuBuffer();

    if (!initialized_) {
      renderer_ = absl::make_unique<mediapipe::QuadRenderer>();
      MP_RETURN_IF_ERROR(renderer_->GlSetup());
      initialized_ = true;
    }

    auto src = helper_.CreateSourceTexture(input);

    EGLSurface old_surface = eglGetCurrentSurface(EGL_DRAW);
    EGLDisplay display = eglGetCurrentDisplay();
    EGLContext context = eglGetCurrentContext();

    // Note that eglMakeCurrent can be very slow on Android if you use it to
    // change the current context, but it is fast if you only change the
    // current surface.
    EGLBoolean success = eglMakeCurrent(display, surface, surface, context);
    RET_CHECK(success) << "failed to make surface current";

    EGLint dst_width;
    success = eglQuerySurface(display, surface, EGL_WIDTH, &dst_width);
    RET_CHECK(success) << "failed to query surface width";

    EGLint dst_height;
    success = eglQuerySurface(display, surface, EGL_HEIGHT, &dst_height);
    RET_CHECK(success) << "failed to query surface height";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, dst_width, dst_height);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src.target(), src.name());

    MP_RETURN_IF_ERROR(
        renderer_->GlRender(src.width(), src.height(), dst_width, dst_height,
                            scale_mode_, mediapipe::FrameRotation::kNone,
                            /*flip_horizontal=*/false, /*flip_vertical=*/false,
                            /*flip_texture=*/surface_holder_->flip_y));

    glBindTexture(src.target(), 0);

    success = eglSwapBuffers(display, surface);
    RET_CHECK(success) << "failed to swap buffers";

    success = eglMakeCurrent(display, old_surface, old_surface, context);
    RET_CHECK(success) << "failed to restore old surface";

    src.Release();
    return absl::OkStatus();
  });
}

GlSurfaceSinkCalculator::~GlSurfaceSinkCalculator() {
  if (renderer_) {
    // TODO: use move capture when we have C++14 or better.
    mediapipe::QuadRenderer* renderer = renderer_.release();
    helper_.RunInGlContext([renderer] {
      renderer->GlTeardown();
      delete renderer;
    });
  }
}

}  // namespace api2
}  // namespace mediapipe
