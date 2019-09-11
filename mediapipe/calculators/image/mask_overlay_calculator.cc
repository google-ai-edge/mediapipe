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

#include "mediapipe/calculators/image/mask_overlay_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gl_simple_shaders.h"
#include "mediapipe/gpu/shader_util.h"

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

namespace mediapipe {

using ::mediapipe::MaskOverlayCalculatorOptions_MaskChannel_ALPHA;
using ::mediapipe::MaskOverlayCalculatorOptions_MaskChannel_RED;
using ::mediapipe::MaskOverlayCalculatorOptions_MaskChannel_UNKNOWN;

// Mixes two frames using a third mask frame or constant value.
//
// Inputs:
//   VIDEO:[0,1] (GpuBuffer):
//     Two inputs should be provided.
//   MASK (GpuBuffer):
//     Optional.
//     Where the mask is 0, VIDEO:0 will be used. Where it is 1, VIDEO:1.
//     Intermediate values will blend.
//     If not specified, CONST_MASK float must be present.
//   CONST_MASK (float):
//     Optional.
//     If not specified, MASK GpuBuffer must be present.
//     Similar to MASK GpuBuffer, but applied globally to every pixel.
//
// Outputs:
//   OUTPUT (GpuBuffer):
//     The mix.

class MaskOverlayCalculator : public CalculatorBase {
 public:
  MaskOverlayCalculator() {}
  ~MaskOverlayCalculator();

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

  ::mediapipe::Status GlSetup(
      const MaskOverlayCalculatorOptions::MaskChannel mask_channel);
  ::mediapipe::Status GlRender(const float mask_const);

 private:
  GlCalculatorHelper helper_;
  bool initialized_ = false;
  bool use_mask_tex_ = false;  // Otherwise, use constant float value.
  GLuint program_ = 0;
  GLint unif_frame1_;
  GLint unif_frame2_;
  GLint unif_mask_;
};
REGISTER_CALCULATOR(MaskOverlayCalculator);

// static
::mediapipe::Status MaskOverlayCalculator::GetContract(CalculatorContract* cc) {
  MP_RETURN_IF_ERROR(GlCalculatorHelper::UpdateContract(cc));
  cc->Inputs().Get("VIDEO", 0).Set<GpuBuffer>();
  cc->Inputs().Get("VIDEO", 1).Set<GpuBuffer>();
  if (cc->Inputs().HasTag("MASK"))
    cc->Inputs().Tag("MASK").Set<GpuBuffer>();
  else if (cc->Inputs().HasTag("CONST_MASK"))
    cc->Inputs().Tag("CONST_MASK").Set<float>();
  else
    return ::mediapipe::Status(
        ::mediapipe::StatusCode::kNotFound,
        "At least one mask input stream must be present.");
  cc->Outputs().Tag("OUTPUT").Set<GpuBuffer>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MaskOverlayCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  if (cc->Inputs().HasTag("MASK")) {
    use_mask_tex_ = true;
  }
  return helper_.Open(cc);
}

::mediapipe::Status MaskOverlayCalculator::Process(CalculatorContext* cc) {
  return helper_.RunInGlContext([this, &cc]() -> ::mediapipe::Status {
    if (!initialized_) {
      const auto& options = cc->Options<MaskOverlayCalculatorOptions>();
      const auto mask_channel = options.mask_channel();

      MP_RETURN_IF_ERROR(GlSetup(mask_channel));
      initialized_ = true;
    }

    glDisable(GL_BLEND);

    const Packet& input1_packet = cc->Inputs().Get("VIDEO", 1).Value();
    const Packet& mask_packet = use_mask_tex_
                                    ? cc->Inputs().Tag("MASK").Value()
                                    : cc->Inputs().Tag("CONST_MASK").Value();

    if (mask_packet.IsEmpty()) {
      cc->Outputs().Tag("OUTPUT").AddPacket(input1_packet);
      return ::mediapipe::OkStatus();
    }

    const auto& input0_buffer = cc->Inputs().Get("VIDEO", 0).Get<GpuBuffer>();
    const auto& input1_buffer = input1_packet.Get<GpuBuffer>();

    auto src1 = helper_.CreateSourceTexture(input0_buffer);
    auto src2 = helper_.CreateSourceTexture(input1_buffer);

    GlTexture mask_tex;
    if (use_mask_tex_) {
      const auto& mask_buffer = mask_packet.Get<GpuBuffer>();
      mask_tex = helper_.CreateSourceTexture(mask_buffer);
    }

    auto dst = helper_.CreateDestinationTexture(src1.width(), src1.height());

    helper_.BindFramebuffer(dst);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src1.target(), src1.name());

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(src2.target(), src2.name());

    if (use_mask_tex_) {
      const float mask_const = -1;

      glActiveTexture(GL_TEXTURE3);
      glBindTexture(mask_tex.target(), mask_tex.name());

      MP_RETURN_IF_ERROR(GlRender(mask_const));

      glActiveTexture(GL_TEXTURE3);
      glBindTexture(mask_tex.target(), 0);

    } else {
      const float mask_const = mask_packet.Get<float>();

      MP_RETURN_IF_ERROR(GlRender(mask_const));
    }

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(src2.target(), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(src1.target(), 0);

    glFlush();

    auto output = dst.GetFrame<GpuBuffer>();
    src1.Release();
    src2.Release();
    if (use_mask_tex_) mask_tex.Release();
    dst.Release();

    cc->Outputs().Tag("OUTPUT").Add(output.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  });
}

::mediapipe::Status MaskOverlayCalculator::GlSetup(
    const MaskOverlayCalculatorOptions::MaskChannel mask_channel) {
  // Load vertex and fragment shaders
  const GLint attr_location[NUM_ATTRIBUTES] = {
      ATTRIB_VERTEX,
      ATTRIB_TEXTURE_POSITION,
  };
  const GLchar* attr_name[NUM_ATTRIBUTES] = {
      "position",
      "texture_coordinate",
  };

  std::string mask_component;
  switch (mask_channel) {
    case MaskOverlayCalculatorOptions_MaskChannel_UNKNOWN:
    case MaskOverlayCalculatorOptions_MaskChannel_RED:
      mask_component = "r";
      break;
    case MaskOverlayCalculatorOptions_MaskChannel_ALPHA:
      mask_component = "a";
      break;
  }

  const std::string frag_src_tex =
      std::string(kMediaPipeFragmentShaderPreamble) +
      R"(
    DEFAULT_PRECISION(highp, float)

    in vec2 sample_coordinate;
    uniform sampler2D frame1;
    uniform sampler2D frame2;
    uniform sampler2D mask;

    void main() {
      vec4 color1 = texture2D(frame1, sample_coordinate);
      vec4 color2 = texture2D(frame2, sample_coordinate);
      vec4 weight = texture2D(mask, sample_coordinate);

    #define MASK_COMPONENT )" +
      mask_component +
      R"(

      gl_FragColor = mix(color1, color2, weight.MASK_COMPONENT);
    }
  )";

  const GLchar* frag_src_const = R"(
    precision highp float;

    varying vec2 sample_coordinate;
    uniform sampler2D frame1;
    uniform sampler2D frame2;
    uniform float mask;

    void main() {
      vec4 color1 = texture2D(frame1, sample_coordinate);
      vec4 color2 = texture2D(frame2, sample_coordinate);
      float weight = mask;

      gl_FragColor = mix(color1, color2, weight);
    }
  )";

  // shader program
  GlhCreateProgram(kBasicVertexShader,
                   use_mask_tex_ ? frag_src_tex.c_str() : frag_src_const,
                   NUM_ATTRIBUTES, &attr_name[0], attr_location, &program_);
  RET_CHECK(program_) << "Problem initializing the program.";
  unif_frame1_ = glGetUniformLocation(program_, "frame1");
  unif_frame2_ = glGetUniformLocation(program_, "frame2");
  unif_mask_ = glGetUniformLocation(program_, "mask");
  return ::mediapipe::OkStatus();
}

::mediapipe::Status MaskOverlayCalculator::GlRender(const float mask_const) {
  glUseProgram(program_);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, kBasicSquareVertices);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0,
                        kBasicTextureVertices);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);

  glUniform1i(unif_frame1_, 1);
  glUniform1i(unif_frame2_, 2);
  if (use_mask_tex_)
    glUniform1i(unif_mask_, 3);
  else
    glUniform1f(unif_mask_, mask_const);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  return ::mediapipe::OkStatus();
}

MaskOverlayCalculator::~MaskOverlayCalculator() {
  helper_.RunInGlContext([this] {
    if (program_) {
      glDeleteProgram(program_);
      program_ = 0;
    }
  });
}

}  // namespace mediapipe
