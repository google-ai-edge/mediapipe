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

#include "mediapipe/calculators/util/thresholding_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"

namespace mediapipe {

constexpr char kThresholdTag[] = "THRESHOLD";
constexpr char kRejectTag[] = "REJECT";
constexpr char kAcceptTag[] = "ACCEPT";
constexpr char kFlagTag[] = "FLAG";
constexpr char kFloatTag[] = "FLOAT";

// Applies a threshold on a stream of numeric values and outputs a flag and/or
// accept/reject stream. The threshold can be specified by one of the following:
//   1) Input stream.
//   2) Input side packet.
//   3) Calculator option.
//
// Input:
//  FLOAT: A float, which will be cast to double to be compared with a
//         threshold of double type.
//  THRESHOLD(optional): A double specifying the threshold at current timestamp.
//
// Output:
//   FLAG(optional): A boolean indicating if the input value is larger than the
//                   threshold.
//   ACCEPT(optional): A packet will be sent if the value is larger than the
//                     threshold.
//   REJECT(optional): A packet will be sent if the value is no larger than the
//                     threshold.
//
// Usage example:
// node {
//   calculator: "ThresholdingCalculator"
//   input_stream: "FLOAT:score"
//   output_stream: "ACCEPT:accept"
//   output_stream: "REJECT:reject"
//   options: {
//     [mediapipe.ThresholdingCalculatorOptions.ext] {
//       threshold: 0.1
//     }
//   }
// }
class ThresholdingCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

 private:
  double threshold_{};
};
REGISTER_CALCULATOR(ThresholdingCalculator);

absl::Status ThresholdingCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kFloatTag));
  cc->Inputs().Tag(kFloatTag).Set<float>();

  if (cc->Outputs().HasTag(kFlagTag)) {
    cc->Outputs().Tag(kFlagTag).Set<bool>();
  }
  if (cc->Outputs().HasTag(kAcceptTag)) {
    cc->Outputs().Tag(kAcceptTag).Set<bool>();
  }
  if (cc->Outputs().HasTag(kRejectTag)) {
    cc->Outputs().Tag(kRejectTag).Set<bool>();
  }
  if (cc->Inputs().HasTag(kThresholdTag)) {
    cc->Inputs().Tag(kThresholdTag).Set<double>();
  }
  if (cc->InputSidePackets().HasTag(kThresholdTag)) {
    cc->InputSidePackets().Tag(kThresholdTag).Set<double>();
    RET_CHECK(!cc->Inputs().HasTag(kThresholdTag))
        << "Using both the threshold input side packet and input stream is not "
           "supported.";
  }

  return absl::OkStatus();
}

absl::Status ThresholdingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options =
      cc->Options<::mediapipe::ThresholdingCalculatorOptions>();
  if (options.has_threshold()) {
    RET_CHECK(!cc->Inputs().HasTag(kThresholdTag))
        << "Using both the threshold option and input stream is not supported.";
    RET_CHECK(!cc->InputSidePackets().HasTag(kThresholdTag))
        << "Using both the threshold option and input side packet is not "
           "supported.";
    threshold_ = options.threshold();
  }

  if (cc->InputSidePackets().HasTag(kThresholdTag)) {
    threshold_ = cc->InputSidePackets().Tag(kThresholdTag).Get<double>();
  }
  return absl::OkStatus();
}

absl::Status ThresholdingCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kThresholdTag) &&
      !cc->Inputs().Tag(kThresholdTag).IsEmpty()) {
    threshold_ = cc->Inputs().Tag(kThresholdTag).Get<double>();
  }

  bool accept = false;
  RET_CHECK(!cc->Inputs().Tag(kFloatTag).IsEmpty());
  accept = static_cast<double>(cc->Inputs().Tag(kFloatTag).Get<float>()) >
           threshold_;

  if (cc->Outputs().HasTag(kFlagTag)) {
    cc->Outputs().Tag(kFlagTag).AddPacket(
        MakePacket<bool>(accept).At(cc->InputTimestamp()));
  }

  if (accept && cc->Outputs().HasTag(kAcceptTag)) {
    cc->Outputs()
        .Tag(kAcceptTag)
        .AddPacket(MakePacket<bool>(true).At(cc->InputTimestamp()));
  }
  if (!accept && cc->Outputs().HasTag(kRejectTag)) {
    cc->Outputs()
        .Tag(kRejectTag)
        .AddPacket(MakePacket<bool>(false).At(cc->InputTimestamp()));
  }

  return absl::OkStatus();
}
}  // namespace mediapipe
