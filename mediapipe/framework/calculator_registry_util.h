// Copyright 2019 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_CALCULATOR_REGISTRY_UTIL_H_
#define MEDIAPIPE_FRAMEWORK_CALCULATOR_REGISTRY_UTIL_H_

#include <memory>

#include "mediapipe/framework/calculator_base.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_state.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/framework/tool/tag_map.h"

// Calculator registry util functions that supports both legacy Calculator API
// and CalculatorBase.
namespace mediapipe {

bool IsLegacyCalculator(const std::string& package_name,
                        const std::string& node_class);

::mediapipe::Status VerifyCalculatorWithContract(
    const std::string& package_name, const std::string& node_class,
    CalculatorContract* contract);

::mediapipe::StatusOr<std::unique_ptr<CalculatorBase>> CreateCalculator(
    const std::shared_ptr<tool::TagMap>& input_tag_map,
    const std::shared_ptr<tool::TagMap>& output_tag_map,
    const std::string& package_name, CalculatorState* calculator_state,
    CalculatorContext* calculator_context);

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_CALCULATOR_REGISTRY_UTIL_H_
