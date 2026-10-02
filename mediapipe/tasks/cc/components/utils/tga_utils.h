/* Copyright 2026 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_TGA_UTILS_H_
#define MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_TGA_UTILS_H_

#include <string>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/image.h"

namespace mediapipe::tasks::components::utils {

// Encodes a `mediapipe::Image` (1, 3, or 4 channels) into an uncompressed TGA
// byte buffer suitable for LiteRT-LM `InputImage` and multimodal task inputs.
absl::StatusOr<std::string> EncodeToTga(const mediapipe::Image& image);

}  // namespace mediapipe::tasks::components::utils

#endif  // MEDIAPIPE_TASKS_CC_COMPONENTS_UTILS_TGA_UTILS_H_
