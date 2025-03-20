/* Copyright 2024 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_TEST_TEST_UTILS_H_
#define MEDIAPIPE_TASKS_C_TEST_TEST_UTILS_H_

#include <string>

#include "absl/flags/flag.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/c/vision/core/common.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace mediapipe::tasks::c::test {

MpMask CreateCategoryMaskFromImage(absl::StatusOr<Image>& image);

float SimilarToUint8Mask(const MpMask* actual_mask, const MpMask* expected_mask,
                         int magnification_factor);

}  // namespace mediapipe::tasks::c::test

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_TEST_TEST_UTILS_H_
