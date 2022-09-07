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

#ifndef MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_UTILS_H_
#define MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_UTILS_H_

#include "mediapipe/calculators/tensor/inference_calculator.pb.h"

namespace mediapipe {

// Returns number of threads to configure XNNPACK delegate with.
// Returns user provided value if specified. Otherwise, tries to choose optimal
// number of threads depending on the device.
int GetXnnpackNumThreads(
    const bool opts_has_delegate,
    const mediapipe::InferenceCalculatorOptions::Delegate& opts_delegate);

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_TENSOR_INFERENCE_CALCULATOR_UTILS_H_
