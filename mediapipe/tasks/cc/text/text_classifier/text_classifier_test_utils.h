/* Copyright 2022 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_TEST_UTILS_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_TEST_UTILS_H_

#include <memory>

#include "tensorflow/lite/mutable_op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace text {

// Create a custom MutableOpResolver to provide custom OP implementations to
// mimic classification behavior.
std::unique_ptr<tflite::MutableOpResolver> CreateCustomResolver();

}  // namespace text
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_CLASSIFIER_TEXT_CLASSIFIER_TEST_UTILS_H_
