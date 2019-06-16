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

#include <utility>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Given two input streams (A, B), output a single stream containing a pair<A,
// B>.
//
// Example config:
// node {
//   calculator: "MakePairCalculator"
//   input_stream: "packet_a"
//   input_stream: "packet_b"
//   output_stream: "output_pair_a_b"
// }
class MakePairCalculator : public CalculatorBase {
 public:
  MakePairCalculator() {}
  ~MakePairCalculator() override {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).SetAny();
    cc->Inputs().Index(1).SetAny();
    cc->Outputs().Index(0).Set<std::pair<Packet, Packet>>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).Add(
        new std::pair<Packet, Packet>(cc->Inputs().Index(0).Value(),
                                      cc->Inputs().Index(1).Value()),
        cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(MakePairCalculator);

}  // namespace mediapipe
