// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"
#include "tensorflow/core/example/example.pb.h"

// A calculator to serialize/deserialize tensorflow::SequenceExample protos
// to and from strings.
//
// Example converting to SequenceExample in Open():
// node {
//   calculator: "StringToSequenceExampleCalculator"
//   input_side_packet: "STRING:serialized_sequence_example"
//   output_side_packet: "SEQUENCE_EXAMPLE:sequence_example"
// }
//
// Example converting to string in Close():
// node {
//   calculator: "StringToSequenceExampleCalculator"
//   input_side_packet: "SEQUENCE_EXAMPLE:sequence_example"
//   output_side_packet: "STRING:serialized_sequence_example"
// }

namespace mediapipe {
namespace tf = ::tensorflow;
namespace {
constexpr char kString[] = "STRING";
constexpr char kSequenceExample[] = "SEQUENCE_EXAMPLE";
}  // namespace

class StringToSequenceExampleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(StringToSequenceExampleCalculator);

absl::Status StringToSequenceExampleCalculator::GetContract(
    CalculatorContract* cc) {
  if (cc->InputSidePackets().HasTag(kString)) {
    cc->InputSidePackets().Tag(kString).Set<std::string>();
    cc->OutputSidePackets().Tag(kSequenceExample).Set<tf::SequenceExample>();
  }
  if (cc->InputSidePackets().HasTag(kSequenceExample)) {
    cc->InputSidePackets().Tag(kSequenceExample).Set<tf::SequenceExample>();
    cc->OutputSidePackets().Tag(kString).Set<std::string>();
  }
  return absl::OkStatus();
}

absl::Status StringToSequenceExampleCalculator::Open(CalculatorContext* cc) {
  if (cc->InputSidePackets().HasTag(kString)) {
    auto string_value = cc->InputSidePackets().Tag(kString).Get<std::string>();
    auto example = absl::make_unique<tf::SequenceExample>();
    example->ParseFromString(string_value);
    cc->OutputSidePackets()
        .Tag(kSequenceExample)
        .Set(mediapipe::Adopt(example.release()));
  }
  return absl::OkStatus();
}

absl::Status StringToSequenceExampleCalculator::Process(CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status StringToSequenceExampleCalculator::Close(CalculatorContext* cc) {
  if (cc->InputSidePackets().HasTag(kSequenceExample)) {
    const auto& example =
        cc->InputSidePackets().Tag(kSequenceExample).Get<tf::SequenceExample>();
    auto string_value = absl::make_unique<std::string>();
    example.SerializeToString(string_value.get());
    cc->OutputSidePackets().Tag(kString).Set(
        mediapipe::Adopt(string_value.release()));
  }
  return absl::OkStatus();
}

}  // namespace mediapipe
