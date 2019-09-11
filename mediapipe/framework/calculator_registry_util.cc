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

#include "mediapipe/framework/calculator_registry_util.h"

#include <algorithm>
#include <string>

#include "mediapipe/framework/collection.h"
#include "mediapipe/framework/collection_item_id.h"
#include "mediapipe/framework/packet_set.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/timestamp.h"

namespace mediapipe {

bool IsLegacyCalculator(const std::string& package_name,
                        const std::string& node_class) {
  return false;
}

::mediapipe::Status VerifyCalculatorWithContract(
    const std::string& package_name, const std::string& node_class,
    CalculatorContract* contract) {
  // A number of calculators use the non-CC methods on GlCalculatorHelper
  // even though they are CalculatorBase-based.
  ASSIGN_OR_RETURN(
      auto static_access_to_calculator_base,
      internal::StaticAccessToCalculatorBaseRegistry::CreateByNameInNamespace(
          package_name, node_class),
      _ << "Unable to find Calculator \"" << node_class << "\"");
  MP_RETURN_IF_ERROR(static_access_to_calculator_base->GetContract(contract))
          .SetPrepend()
      << node_class << ": ";
  return ::mediapipe::OkStatus();
}

::mediapipe::StatusOr<std::unique_ptr<CalculatorBase>> CreateCalculator(
    const std::shared_ptr<tool::TagMap>& input_tag_map,
    const std::shared_ptr<tool::TagMap>& output_tag_map,
    const std::string& package_name, CalculatorState* calculator_state,
    CalculatorContext* calculator_context) {
  std::unique_ptr<CalculatorBase> calculator;
  ASSIGN_OR_RETURN(calculator,
                   CalculatorBaseRegistry::CreateByNameInNamespace(
                       package_name, calculator_state->CalculatorType()));
  return std::move(calculator);
}

}  // namespace mediapipe
