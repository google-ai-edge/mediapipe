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

#include "mediapipe/tasks/cc/text/text_embedder/text_embedder_test_utils.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/shims/cc/kernels/register.h"

namespace tflite::ops::custom {
TfLiteRegistration* Register_SENTENCEPIECE_TOKENIZER();
TfLiteRegistration* Register_RAGGED_TENSOR_TO_TENSOR();
}  // namespace tflite::ops::custom

namespace mediapipe::tasks::text::text_embedder {

std::unique_ptr<tflite::OpResolver> CreateUSEOpResolver() {
  auto resolver =
      absl::make_unique<tflite_shims::ops::builtin::BuiltinOpResolver>();
  resolver->AddCustom(
      "TFSentencepieceTokenizeOp",
      ::tflite::ops::custom::Register_SENTENCEPIECE_TOKENIZER());
  resolver->AddCustom(
      "RaggedTensorToTensor",
      ::tflite::ops::custom::Register_RAGGED_TENSOR_TO_TENSOR());
  return resolver;
}

}  // namespace mediapipe::tasks::text::text_embedder
