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

#include "mediapipe/util/frame_buffer/buffer_common.h"

namespace mediapipe {
namespace frame_buffer {
namespace common {

bool crop_buffer(int x0, int y0, int x1, int y1, halide_buffer_t* buffer) {
  if (x0 < 0 || x1 >= buffer->dim[0].extent) {
    return false;
  }
  if (y0 < 0 || y1 >= buffer->dim[1].extent) {
    return false;
  }

  // Move the start pointer so that it points at (x0, y0) and set the new
  // extents. Leave the strides unchanged; we simply skip over the cropped
  // image data.
  buffer->host += y0 * buffer->dim[1].stride + x0 * buffer->dim[0].stride;
  buffer->dim[0].extent = x1 - x0 + 1;
  buffer->dim[1].extent = y1 - y0 + 1;
  return true;
}

}  // namespace common
}  // namespace frame_buffer
}  // namespace mediapipe
