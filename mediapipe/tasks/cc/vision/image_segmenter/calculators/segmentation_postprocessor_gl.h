// Copyright 2023 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SEGMENTATION_POSTPROCESSOR_GL_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SEGMENTATION_POSTPROCESSOR_GL_H_
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_utils.h"

// On Android with compute shaders we include the SSBO-to-texture converter
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31 && \
    defined(__ANDROID__)
#define TASK_SEGMENTATION_USE_GLES_31_POSTPROCESSING 1
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/ssbo_to_texture_converter.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31 &&
        // defined(__ANDROID__)

namespace mediapipe {
namespace tasks {

class SegmentationPostprocessorGl {
 public:
  ~SegmentationPostprocessorGl();

  static absl::Status UpdateContract(CalculatorContract* cc);

  absl::Status Initialize(
      CalculatorContext* cc,
      TensorsToSegmentationCalculatorOptions const& options);
  std::vector<std::unique_ptr<Image>> GetSegmentationResultGpu(
      const vision::Shape& input_shape, const vision::Shape& output_shape,
      const Tensor& tensor, const bool produce_confidence_masks,
      const bool produce_category_mask);

 private:
  struct GlShader {
    GLuint program = 0;
    absl::flat_hash_map<std::string, GLint> uniforms;
  };

  absl::Status GlInit(const bool produce_confidence_masks);
  bool HasGlExtension(std::string const& extension);
  absl::Status CreateBasicFragmentShaderProgram(
      std::string const& program_name,
      std::string const& fragment_shader_source,
      std::vector<std::string> const& uniform_names,
      GlShader* shader_struct_ptr, bool is_es30_only);

  TensorsToSegmentationCalculatorOptions options_;
  GlCalculatorHelper helper_;

  // GL references (programs, buffers, uniforms)
  // Split program is special because it uses a custom vertex shader.
  GLuint split_program_ = 0;
  GLuint square_vertices_ = 0;
  GLuint texture_vertices_ = 0;
  GLint split_texture_uniform_;
  GLint split_x_offset_uniform_;

  GlShader activation_shader_;
  GlShader argmax_shader_;
  GlShader argmax_one_class_shader_;
  GlShader channel_select_shader_;
  GlShader softmax_max_shader_;
  GlShader softmax_transform_and_sum_shader_;
  GlShader softmax_normalization_shader_;

#ifdef TASK_SEGMENTATION_USE_GLES_31_POSTPROCESSING
  SsboToTextureConverter ssbo_to_texture_converter_;
#endif
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SEGMENTATION_POSTPROCESSOR_GL_H_
