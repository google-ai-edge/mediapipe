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

#include "mediapipe/calculators/tensor/tensors_to_segmentation_utils.h"

#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#if !MEDIAPIPE_DISABLE_GPU
#include "mediapipe/gpu/gl_base.h"
#endif  // !MEDIAPIPE_DISABLE_GPU

namespace {
enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };
}  // namespace

namespace mediapipe {
namespace tensors_to_segmentation_utils {

int NumGroups(int size, int group_size) {
  return (size + group_size - 1) / group_size;
}

bool CanUseGpu() {
#if !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
  // TODO: Configure GPU usage policy in individual calculators.
  constexpr bool kAllowGpuProcessing = true;
  return kAllowGpuProcessing;
#else
  return false;
#endif  // !MEDIAPIPE_DISABLE_GPU || MEDIAPIPE_METAL_ENABLED
}

absl::StatusOr<std::tuple<int, int, int>> GetHwcFromDims(
    const std::vector<int>& dims) {
  if (dims.size() == 3) {
    return std::make_tuple(dims[0], dims[1], dims[2]);
  } else if (dims.size() == 4) {
    // BHWC format check B == 1
    RET_CHECK_EQ(dims[0], 1) << "Expected batch to be 1 for BHWC heatmap";
    return std::make_tuple(dims[1], dims[2], dims[3]);
  } else {
    RET_CHECK(false) << "Invalid shape for segmentation tensor " << dims.size();
  }
}

void GlRender() {
#if !MEDIAPIPE_DISABLE_GPU
  static const GLfloat square_vertices[] = {
      -1.0f, -1.0f,  // bottom left
      1.0f,  -1.0f,  // bottom right
      -1.0f, 1.0f,   // top left
      1.0f,  1.0f,   // top right
  };
  static const GLfloat texture_vertices[] = {
      0.0f, 0.0f,  // bottom left
      1.0f, 0.0f,  // bottom right
      0.0f, 1.0f,  // top left
      1.0f, 1.0f,  // top right
  };

  // vertex storage
  GLuint vbo[2];
  glGenBuffers(2, vbo);
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  // vbo 0
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), square_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_VERTEX);
  glVertexAttribPointer(ATTRIB_VERTEX, 2, GL_FLOAT, 0, 0, nullptr);

  // vbo 1
  glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
  glBufferData(GL_ARRAY_BUFFER, 4 * 2 * sizeof(GLfloat), texture_vertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glVertexAttribPointer(ATTRIB_TEXTURE_POSITION, 2, GL_FLOAT, 0, 0, nullptr);

  // draw
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // cleanup
  glDisableVertexAttribArray(ATTRIB_VERTEX);
  glDisableVertexAttribArray(ATTRIB_TEXTURE_POSITION);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(2, vbo);
#endif  // !MEDIAPIPE_DISABLE_GPU
}

}  // namespace tensors_to_segmentation_utils
}  // namespace mediapipe
