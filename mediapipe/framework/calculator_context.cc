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

#include "mediapipe/framework/calculator_context.h"

#include "absl/log/absl_check.h"

namespace mediapipe {

const std::string& CalculatorContext::CalculatorType() const {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->CalculatorType();
}

const CalculatorOptions& CalculatorContext::Options() const {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->Options();
}

const std::string& CalculatorContext::NodeName() const {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->NodeName();
}

int CalculatorContext::NodeId() const {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->NodeId();
}

Counter* CalculatorContext::GetCounter(const std::string& name) {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->GetCounter(name);
}

CounterFactory* CalculatorContext::GetCounterFactory() {
  ABSL_CHECK(calculator_state_);
  return calculator_state_->GetCounterFactory();
}

const PacketSet& CalculatorContext::InputSidePackets() const {
  return calculator_state_->InputSidePackets();
}

OutputSidePacketSet& CalculatorContext::OutputSidePackets() {
  return calculator_state_->OutputSidePackets();
}

InputStreamShardSet& CalculatorContext::Inputs() { return inputs_; }

const InputStreamShardSet& CalculatorContext::Inputs() const { return inputs_; }

OutputStreamShardSet& CalculatorContext::Outputs() { return outputs_; }

const OutputStreamShardSet& CalculatorContext::Outputs() const {
  return outputs_;
}

void CalculatorContext::SetOffset(TimestampDiff offset) {
  for (auto& stream : outputs_) {
    stream.SetOffset(offset);
  }
}

const InputStreamSet& CalculatorContext::InputStreams() const {
  if (!input_streams_) {
    input_streams_ = absl::make_unique<InputStreamSet>(inputs_.TagMap());
    for (CollectionItemId id = input_streams_->BeginId();
         id < input_streams_->EndId(); ++id) {
      input_streams_->Get(id) = const_cast<InputStreamShard*>(&inputs_.Get(id));
    }
  }
  return *input_streams_;
}

const OutputStreamSet& CalculatorContext::OutputStreams() const {
  if (!output_streams_) {
    output_streams_ = absl::make_unique<OutputStreamSet>(outputs_.TagMap());
    for (CollectionItemId id = output_streams_->BeginId();
         id < output_streams_->EndId(); ++id) {
      output_streams_->Get(id) =
          const_cast<OutputStreamShard*>(&outputs_.Get(id));
    }
  }
  return *output_streams_;
}

}  // namespace mediapipe
