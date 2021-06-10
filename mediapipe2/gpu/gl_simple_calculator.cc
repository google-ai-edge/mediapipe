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

#include "mediapipe/gpu/gl_simple_calculator.h"

namespace mediapipe {

// static
absl::Status GlSimpleCalculator::GetContract(CalculatorContract* cc) {
  TagOrIndex(&cc->Inputs(), "VIDEO", 0).Set<GpuBuffer>();
  TagOrIndex(&cc->Outputs(), "VIDEO", 0).Set<GpuBuffer>();
  // Currently we pass GL context information and other stuff as external
  // inputs, which are handled by the helper.
  return GlCalculatorHelper::UpdateContract(cc);
}

absl::Status GlSimpleCalculator::Open(CalculatorContext* cc) {
  // Inform the framework that we always output at the same timestamp
  // as we receive a packet at.
  cc->SetOffset(mediapipe::TimestampDiff(0));

  // Let the helper access the GL context information.
  return helper_.Open(cc);
}

absl::Status GlSimpleCalculator::Process(CalculatorContext* cc) {
  return RunInGlContext([this, cc]() -> absl::Status {
    const auto& input = TagOrIndex(cc->Inputs(), "VIDEO", 0).Get<GpuBuffer>();
    if (!initialized_) {
      MP_RETURN_IF_ERROR(GlSetup());
      initialized_ = true;
    }

    auto src = helper_.CreateSourceTexture(input);
    int dst_width;
    int dst_height;
    GetOutputDimensions(src.width(), src.height(), &dst_width, &dst_height);
    auto dst = helper_.CreateDestinationTexture(dst_width, dst_height,
                                                GetOutputFormat());

    helper_.BindFramebuffer(dst);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src.target(), src.name());

    MP_RETURN_IF_ERROR(GlBind());
    // Run core program.
    MP_RETURN_IF_ERROR(GlRender(src, dst));

    glBindTexture(src.target(), 0);

    glFlush();

    auto output = dst.GetFrame<GpuBuffer>();

    src.Release();
    dst.Release();

    TagOrIndex(&cc->Outputs(), "VIDEO", 0)
        .Add(output.release(), cc->InputTimestamp());

    return absl::OkStatus();
  });
}

absl::Status GlSimpleCalculator::Close(CalculatorContext* cc) {
  return RunInGlContext([this]() -> absl::Status { return GlTeardown(); });
}

}  // namespace mediapipe
