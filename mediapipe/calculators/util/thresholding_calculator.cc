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
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  double threshold_{};
};
REGISTER_CALCULATOR(ThresholdingCalculator);

::mediapipe::Status ThresholdingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("FLOAT"));
  cc->Inputs().Tag("FLOAT").Set<float>();

  if (cc->Outputs().HasTag("FLAG")) {
    cc->Outputs().Tag("FLAG").Set<bool>();
  }
  if (cc->Outputs().HasTag("ACCEPT")) {
    cc->Outputs().Tag("ACCEPT").Set<bool>();
  }
  if (cc->Outputs().HasTag("REJECT")) {
    cc->Outputs().Tag("REJECT").Set<bool>();
  }
  if (cc->Inputs().HasTag("THRESHOLD")) {
    cc->Inputs().Tag("THRESHOLD").Set<double>();
  }
  if (cc->InputSidePackets().HasTag("THRESHOLD")) {
    cc->InputSidePackets().Tag("THRESHOLD").Set<double>();
    RET_CHECK(!cc->Inputs().HasTag("THRESHOLD"))
        << "Using both the threshold input side packet and input stream is not "
           "supported.";
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status ThresholdingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options =
      cc->Options<::mediapipe::ThresholdingCalculatorOptions>();
  if (options.has_threshold()) {
    RET_CHECK(!cc->Inputs().HasTag("THRESHOLD"))
        << "Using both the threshold option and input stream is not supported.";
    RET_CHECK(!cc->InputSidePackets().HasTag("THRESHOLD"))
        << "Using both the threshold option and input side packet is not "
           "supported.";
    threshold_ = options.threshold();
  }

  if (cc->InputSidePackets().HasTag("THRESHOLD")) {
    threshold_ = cc->InputSidePackets().Tag("THRESHOLD").Get<float>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status ThresholdingCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag("THRESHOLD") &&
      !cc->Inputs().Tag("THRESHOLD").IsEmpty()) {
    threshold_ = cc->Inputs().Tag("THRESHOLD").Get<double>();
  }

  bool accept = false;
  RET_CHECK(!cc->Inputs().Tag("FLOAT").IsEmpty());
  accept =
      static_cast<double>(cc->Inputs().Tag("FLOAT").Get<float>()) > threshold_;

  if (cc->Outputs().HasTag("FLAG")) {
    cc->Outputs().Tag("FLAG").AddPacket(
        MakePacket<bool>(accept).At(cc->InputTimestamp()));
  }

  if (accept && cc->Outputs().HasTag("ACCEPT")) {
    cc->Outputs().Tag("ACCEPT").AddPacket(
        MakePacket<bool>(true).At(cc->InputTimestamp()));
  }
  if (!accept && cc->Outputs().HasTag("REJECT")) {
    cc->Outputs().Tag("REJECT").AddPacket(
        MakePacket<bool>(false).At(cc->InputTimestamp()));
  }

  return ::mediapipe::OkStatus();
}
}  // namespace mediapipe
