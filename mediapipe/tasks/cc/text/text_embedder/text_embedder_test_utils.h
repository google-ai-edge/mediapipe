/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#ifndef MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_TEST_UTILS_H_
#define MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_TEST_UTILS_H_

#include <memory>

#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe::tasks::text::text_embedder {

// Creates a custom OpResolver containing the additional SENTENCEPIECE_TOKENIZER
// and RAGGED_TENSOR_TO_TENSOR ops needed by universal sentence encoder-based
// models.
std::unique_ptr<tflite::OpResolver> CreateUSEOpResolver();

}  // namespace mediapipe::tasks::text::text_embedder

#endif  // MEDIAPIPE_TASKS_CC_TEXT_TEXT_EMBEDDER_TEXT_EMBEDDER_TEST_UTILS_H_
