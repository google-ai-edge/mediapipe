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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

// Attach the header from one stream to another stream.
//
// The header stream (tag HEADER) must not have any packets in it.
//
// Before using this calculator, please think about changing your
// calculator to not need a header or to accept a separate stream with
// a header, that would be more future proof.
//
class AddHeaderCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("HEADER").SetNone();
    cc->Inputs().Tag("DATA").SetAny();
    cc->Outputs().Index(0).SetSameAs(&cc->Inputs().Tag("DATA"));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    const Packet& header = cc->Inputs().Tag("HEADER").Header();
    if (!header.IsEmpty()) {
      cc->Outputs().Index(0).SetHeader(header);
    }
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    cc->Outputs().Index(0).AddPacket(cc->Inputs().Tag("DATA").Value());
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(AddHeaderCalculator);
}  // namespace mediapipe
