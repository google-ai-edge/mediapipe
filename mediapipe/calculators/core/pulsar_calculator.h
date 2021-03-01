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

#ifndef MEDIAPIPE_CALCULATORS_CORE_PULSAR_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_CORE_PULSAR_CALCULATOR_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

// This turns a packet of vector<NormalizedRect> into as many NormalizedRect
// items there are in said vector.
template <typename T>
class PulsarCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK_EQ(cc->Inputs().NumEntries(), 1);
    RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);
    cc->Inputs().Index(0).Set<std::vector<T>>();
    cc->Outputs().Index(0).Set<T>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) final {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) final {
    const auto& items = cc->Inputs().Index(0).Get<std::vector<T>>();
    auto ts = std::max(cc->InputTimestamp(), ts_);
    for (const auto& item : items) {
      auto just_one = absl::make_unique<T>(item);
      cc->Outputs().Index(0).Add(just_one.release(), ts++);
    }
    ts_ = ts;
    return ::mediapipe::OkStatus();
  }

private:
  Timestamp ts_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_CORE_PULSAR_CALCULATOR_H_
