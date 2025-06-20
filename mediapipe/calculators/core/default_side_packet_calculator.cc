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

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kOptionalValueTag[] = "OPTIONAL_VALUE";
constexpr char kDefaultValueTag[] = "DEFAULT_VALUE";
constexpr char kValueTag[] = "VALUE";

}  // namespace

// Outputs side packet default value if optional value is not provided.
//
// This calculator utilizes the fact that MediaPipe automatically removes
// optional side packets of the calculator configuration (i.e. OPTIONAL_VALUE).
// And if it happens - returns default value, otherwise - returns optional
// value.
//
// Input:
//   OPTIONAL_VALUE (optional) - AnyType (but same type as DEFAULT_VALUE)
//     Optional side packet value that is outputted by the calculator as is if
//     provided.
//
//   DEFAULT_VALUE - AnyType
//     Default side pack value that is outputted by the calculator if
//     OPTIONAL_VALUE is not provided.
//
// Output:
//   VALUE - AnyType (but same type as DEFAULT_VALUE)
//     Either OPTIONAL_VALUE (if provided) or DEFAULT_VALUE (otherwise).
//
// Usage example:
//   node {
//     calculator: "DefaultSidePacketCalculator"
//     input_side_packet: "OPTIONAL_VALUE:segmentation_mask_enabled_optional"
//     input_side_packet: "DEFAULT_VALUE:segmentation_mask_enabled_default"
//     output_side_packet: "VALUE:segmentation_mask_enabled"
//   }
class DefaultSidePacketCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(DefaultSidePacketCalculator);

absl::Status DefaultSidePacketCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->InputSidePackets().HasTag(kDefaultValueTag))
      << "Default value must be provided";
  cc->InputSidePackets().Tag(kDefaultValueTag).SetAny();

  // Optional input side packet can be unspecified. In this case MediaPipe will
  // remove it from the calculator config.
  if (cc->InputSidePackets().HasTag(kOptionalValueTag)) {
    cc->InputSidePackets()
        .Tag(kOptionalValueTag)
        .SetSameAs(&cc->InputSidePackets().Tag(kDefaultValueTag))
        .Optional();
  }

  RET_CHECK(cc->OutputSidePackets().HasTag(kValueTag));
  cc->OutputSidePackets().Tag(kValueTag).SetSameAs(
      &cc->InputSidePackets().Tag(kDefaultValueTag));

  return absl::OkStatus();
}

absl::Status DefaultSidePacketCalculator::Open(CalculatorContext* cc) {
  // If optional value is provided it is returned as the calculator output.
  if (cc->InputSidePackets().HasTag(kOptionalValueTag)) {
    auto& packet = cc->InputSidePackets().Tag(kOptionalValueTag);
    cc->OutputSidePackets().Tag(kValueTag).Set(packet);
    return absl::OkStatus();
  }

  // If no optional value
  auto& packet = cc->InputSidePackets().Tag(kDefaultValueTag);
  cc->OutputSidePackets().Tag(kValueTag).Set(packet);

  return absl::OkStatus();
}

absl::Status DefaultSidePacketCalculator::Process(CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace mediapipe
