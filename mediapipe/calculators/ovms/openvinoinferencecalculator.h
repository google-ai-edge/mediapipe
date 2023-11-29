#pragma once
//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <memory>
#include <unordered_map>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#include "tensorflow/lite/interpreter.h"
#pragma GCC diagnostic pop
class InferenceAdapter;
namespace mediapipe {
class OpenVINOInferenceCalculator : public CalculatorBase {
    std::shared_ptr<::InferenceAdapter> session{nullptr};
    std::unordered_map<std::string, std::string> outputNameToTag;
    std::vector<std::string> input_order_list;
    std::vector<std::string> output_order_list;
    std::unique_ptr<tflite::Interpreter> interpreter_ = absl::make_unique<tflite::Interpreter>();
    bool initialized = false;
public:
    static absl::Status GetContract(CalculatorContract* cc);
    absl::Status Close(CalculatorContext* cc) override final;
    absl::Status Open(CalculatorContext* cc) override final;
    absl::Status Process(CalculatorContext* cc) override final;
};
}  // namespace mediapipe
