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

#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

class SimpleCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Outputs().Index(0).Set<int>();
    if (cc->InputSidePackets().HasTag("MAX_COUNT")) {
      cc->InputSidePackets().Tag("MAX_COUNT").Set<int>();
    }
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) final {
    ABSL_LOG(WARNING) << "Simple Calculator Process called, count_: " << count_;
    int max_count = 1;
    if (cc->InputSidePackets().HasTag("MAX_COUNT")) {
      max_count = cc->InputSidePackets().Tag("MAX_COUNT").Get<int>();
    }
    if (count_ >= max_count) {
      return tool::StatusStop();
    }
    cc->Outputs().Index(0).Add(new int(count_), Timestamp(count_));
    ++count_;
    return absl::OkStatus();
  }

 private:
  int count_ = 0;
};
REGISTER_CALCULATOR(SimpleCalculator);

}  // namespace mediapipe
