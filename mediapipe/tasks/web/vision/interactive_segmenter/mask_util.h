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

#ifndef MEDIAPIPE_TASKS_WEB_VISION_INTERACTIVE_SEGMENTER_MASK_UTIL_H_
#define MEDIAPIPE_TASKS_WEB_VISION_INTERACTIVE_SEGMENTER_MASK_UTIL_H_

#include <cstdint>

#include "absl/status/status.h"

namespace mediapipe::tasks::web::vision {

// Copies row bytes sequentially to copy the C++ mask layout.
// Supports extracting the active channel bytes directly.
//
// Parameters:
// - src_data: The raw source pixel data.
// - width: Width of the mask in pixels.
// - height: Height of the mask in pixels.
// - channels: Number of channels in the source mask (supports 1 or 4 channels).
// - channel_size: Stride of a single channel in bytes (typically 1 byte).
// - width_step: Stride of a single row in bytes (could include alignment
// padding).
// - dest_data: The pre-allocated destination buffer to copy into.
//
// Returns absl::OkStatus() if successful, or absl::InvalidArgumentError if
// arguments are invalid or channels count is unsupported.
//
// Note: This C++ mask utility was originally written for the interactive
// segmenter task.
absl::Status CopyMask(const uint8_t* src_data, int width, int height,
                      int channels, int channel_size, int width_step,
                      uint8_t* dest_data);

}  // namespace mediapipe::tasks::web::vision

#endif  // MEDIAPIPE_TASKS_WEB_VISION_INTERACTIVE_SEGMENTER_MASK_UTIL_H_
