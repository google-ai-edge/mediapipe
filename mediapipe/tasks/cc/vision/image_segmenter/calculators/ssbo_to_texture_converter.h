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

#ifndef MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SSBO_TO_TEXTURE_CONVERTER_H_
#define MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SSBO_TO_TEXTURE_CONVERTER_H_

#include <utility>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/gl_base.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_texture.h"

namespace mediapipe {
namespace tasks {

// Helper class for converting Android and Linux Tensors from OpenGL ES >=3.1
// SSBO objects into OpenGL ES <=3.0 2D textures. Cannot be used with other
// Tensor backends.
class SsboToTextureConverter {
 public:
  SsboToTextureConverter() = default;
  ~SsboToTextureConverter() = default;
  absl::Status Init();
  void Close();
  absl::StatusOr<GLuint> ConvertTensorToGlTexture(const Tensor& tensor,
                                                  const uint32_t width,
                                                  const uint32_t height,
                                                  const uint32_t channels);

  // Should only be called after ConvertTensorToGlTexture
  std::pair<const uint32_t, const uint32_t> GetTextureSize();

 private:
  uint32_t texture_width_;
  uint32_t texture_height_;
  tflite::gpu::gl::GlTexture out_texture_;
  std::unique_ptr<tflite::gpu::gl::GlProgram> delinearization_program_;
};

}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_VISION_IMAGE_SEGMENTER_CALCULATORS_SSBO_TO_TEXTURE_CONVERTER_H_
