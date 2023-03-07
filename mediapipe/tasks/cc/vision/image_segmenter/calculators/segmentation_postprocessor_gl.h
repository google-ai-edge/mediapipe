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
      const Tensor& tensor);

 private:
  absl::Status GlInit();

  TensorsToSegmentationCalculatorOptions options_;
  GlCalculatorHelper helper_;

  // GL references (programs, buffers, uniforms)
  GLuint activation_program_ = 0;
  GLuint argmax_program_ = 0;
  GLuint channel_select_program_ = 0;
  GLuint softmax_program_ = 0;
  GLuint split_program_ = 0;
  GLuint square_vertices_ = 0;
  GLuint texture_vertices_ = 0;
  GLint activation_texture_uniform_;
  GLint argmax_texture0_uniform_;
  GLint argmax_texture1_uniform_;
  GLint argmax_texture2_uniform_;
  GLint channel_select_texture_uniform_;
  GLint channel_select_index_uniform_;
  GLint softmax_texture0_uniform_;
  GLint softmax_texture1_uniform_;
  GLint softmax_texture2_uniform_;
  GLint softmax_chunk_select_uniform_;
  GLint split_texture_uniform_;
  GLint split_x_offset_uniform_;
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SEGMENTATION_POSTPROCESSOR_GL_H_
