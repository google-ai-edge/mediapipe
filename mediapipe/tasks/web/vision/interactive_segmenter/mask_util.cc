// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/tasks/web/vision/interactive_segmenter/mask_util.h"

#include <cstdint>
#include <cstring>

#include "absl/status/status.h"

namespace mediapipe::tasks::web::vision {

absl::Status CopyMask(const uint8_t* src_data, int width, int height,
                      int channels, int channel_size, int width_step,
                      uint8_t* dest_data) {
  if (src_data == nullptr || dest_data == nullptr) {
    return absl::InvalidArgumentError(
        "Source and destination buffers must not be null.");
  }
  if (width <= 0 || height <= 0) {
    return absl::InvalidArgumentError(
        "Width and height must be strictly positive.");
  }
  if (channel_size <= 0) {
    return absl::InvalidArgumentError(
        "Channel size must be strictly positive.");
  }
  if (width_step < width * channels * channel_size) {
    return absl::InvalidArgumentError(
        "Width step is smaller than the expected row size.");
  }

  if (channels == 1) {
    for (int y = 0; y < height; ++y) {
      std::memcpy(dest_data + y * width * channel_size,
                  src_data + y * width_step, width * channel_size);
    }
    return absl::OkStatus();
  } else if (channels == 4) {
    for (int y = 0; y < height; ++y) {
      const uint8_t* row_ptr = src_data + y * width_step;
      uint8_t* dest_row_ptr = dest_data + y * width * channel_size;
      for (int x = 0; x < width; ++x) {
        const uint8_t* pixel_ptr = row_ptr + x * channels * channel_size;
        uint8_t* dest_pixel_ptr = dest_row_ptr + x * channel_size;
        std::memcpy(dest_pixel_ptr, pixel_ptr, channel_size);
      }
    }
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError(
      "Unsupported number of channels in CopyMask.");
}

}  // namespace mediapipe::tasks::web::vision
