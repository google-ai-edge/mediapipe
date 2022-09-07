// Copyright 2022 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/tensor/inference_runner.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"

namespace mediapipe {

using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, std::function<void(TfLiteDelegate*)>>;

// Creates inference runner which run inference using newly initialized
// interpreter and provided `delegate`.
//
// `delegate` can be nullptr, in that case newly initialized interpreter will
// use what is available by default.
absl::StatusOr<std::unique_ptr<InferenceRunner>>
CreateInferenceInterpreterDelegateRunner(
    api2::Packet<TfLiteModelPtr> model,
    api2::Packet<tflite::OpResolver> op_resolver, TfLiteDelegatePtr delegate,
    int interpreter_num_threads);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_INTERPRETER_DELEGATE_RUNNER_H_
