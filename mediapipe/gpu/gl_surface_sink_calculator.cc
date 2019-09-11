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
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/egl_surface_holder.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_quad_renderer.h"
#include "mediapipe/gpu/gl_surface_sink_calculator.pb.h"
#include "mediapipe/gpu/shader_util.h"

namespace mediapipe {

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
class GlSurfaceSinkCalculator : public CalculatorBase {
 public:
  GlSurfaceSinkCalculator() : initialized_(false) {}
  ~GlSurfaceSinkCalculator() override;

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  GlCalculatorHelper helper_;
  EglSurfaceHolder* surface_holder_;
  bool initialized_;
  std::unique_ptr<QuadRenderer> renderer_;
  FrameScaleMode scale_mode_ = FrameScaleMode::kFillAndCrop;
};
REGISTER_CALCULATOR(GlSurfaceSinkCalculator);

// static
::mediapipe::Status GlSurfaceSinkCalculator::GetContract(
    CalculatorContract* cc) {
  TagOrIndex(&(cc->Inputs()), "VIDEO", 0).Set<GpuBuffer>();
  cc->InputSidePackets()
      .Tag("SURFACE")
      .Set<std::unique_ptr<EglSurfaceHolder>>();
  // Currently we pass GL context information and other stuff as external
  // inputs, which are handled by the helper.
  return GlCalculatorHelper::UpdateContract(cc);
}

::mediapipe::Status GlSurfaceSinkCalculator::Open(CalculatorContext* cc) {
  surface_holder_ = cc->InputSidePackets()
                        .Tag("SURFACE")
                        .Get<std::unique_ptr<EglSurfaceHolder>>()
                        .get();

  scale_mode_ = FrameScaleModeFromProto(
      cc->Options<GlSurfaceSinkCalculatorOptions>().frame_scale_mode(),
      FrameScaleMode::kFillAndCrop);

  // Let the helper access the GL context information.
  return helper_.Open(cc);
}

::mediapipe::Status GlSurfaceSinkCalculator::Process(CalculatorContext* cc) {
  return helper_.RunInGlContext([this, &cc]() -> ::mediapipe::Status {
    absl::MutexLock lock(&surface_holder_->mutex);
    EGLSurface surface = surface_holder_->surface;
    if (surface == EGL_NO_SURFACE) {
      LOG_EVERY_N(INFO, 300) << "GlSurfaceSinkCalculator: no surface";
      return ::mediapipe::OkStatus();
    }

    const auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0).Get<GpuBuffer>();
    if (!initialized_) {
      renderer_ = absl::make_unique<QuadRenderer>();
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
                            scale_mode_, FrameRotation::kNone,
                            /*flip_horizontal=*/false, /*flip_vertical=*/false,
                            /*flip_texture=*/surface_holder_->flip_y));

    glBindTexture(src.target(), 0);

    success = eglSwapBuffers(display, surface);
    RET_CHECK(success) << "failed to swap buffers";

    success = eglMakeCurrent(display, old_surface, old_surface, context);
    RET_CHECK(success) << "failed to restore old surface";

    src.Release();
    return ::mediapipe::OkStatus();
  });
}

GlSurfaceSinkCalculator::~GlSurfaceSinkCalculator() {
  if (renderer_) {
    // TODO: use move capture when we have C++14 or better.
    QuadRenderer* renderer = renderer_.release();
    helper_.RunInGlContext([renderer] {
      renderer->GlTeardown();
      delete renderer;
    });
  }
}

}  // namespace mediapipe
