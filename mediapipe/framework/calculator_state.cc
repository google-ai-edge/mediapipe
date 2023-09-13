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

// Definitions for CalculatorNode.

#include "mediapipe/framework/calculator_state.h"

#include <string>

#include "absl/log/absl_check.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

CalculatorState::CalculatorState(
    const std::string& node_name, int node_id,
    const std::string& calculator_type,
    const CalculatorGraphConfig::Node& node_config,
    std::shared_ptr<ProfilingContext> profiling_context)
    : node_name_(node_name),
      node_id_(node_id),
      calculator_type_(calculator_type),
      node_config_(node_config),
      profiling_context_(profiling_context),
      counter_factory_(nullptr) {
  options_.Initialize(node_config);
  ResetBetweenRuns();
}

CalculatorState::~CalculatorState() {}

void CalculatorState::ResetBetweenRuns() {
  input_side_packets_ = nullptr;
  counter_factory_ = nullptr;
}

void CalculatorState::SetInputSidePackets(const PacketSet* input_side_packets) {
  ABSL_CHECK(input_side_packets);
  input_side_packets_ = input_side_packets;
}

void CalculatorState::SetOutputSidePackets(
    OutputSidePacketSet* output_side_packets) {
  ABSL_CHECK(output_side_packets);
  output_side_packets_ = output_side_packets;
}

Counter* CalculatorState::GetCounter(const std::string& name) {
  ABSL_CHECK(counter_factory_);
  return counter_factory_->GetCounter(absl::StrCat(NodeName(), "-", name));
}

CounterFactory* CalculatorState::GetCounterFactory() {
  ABSL_CHECK(counter_factory_);
  return counter_factory_;
}

}  // namespace mediapipe
