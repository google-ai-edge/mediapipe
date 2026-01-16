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

#include <sys/types.h>

#include <cstdint>
#include <memory>
#include <string>

#include "absl/strings/numbers.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

// Calculator that converts a string into an integer type, or fails if the
// conversion is not possible.
//
// Example config:
// node {
//   calculator: "StringToIntCalculator"
//   input_side_packet: "string"
//   output_side_packet: "index"
// }
template <typename IntType>
class StringToIntCalculatorTemplate : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->InputSidePackets().Index(0).Set<std::string>();
    cc->OutputSidePackets().Index(0).Set<IntType>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    IntType number;
    if (!absl::SimpleAtoi(cc->InputSidePackets().Index(0).Get<std::string>(),
                          &number)) {
      return absl::InvalidArgumentError(
          "The string could not be parsed as an integer.");
    }
    cc->OutputSidePackets().Index(0).Set(MakePacket<IntType>(number));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    return absl::OkStatus();
  }
};

using StringToIntCalculator = StringToIntCalculatorTemplate<int>;
REGISTER_CALCULATOR(StringToIntCalculator);

using StringToUintCalculator = StringToIntCalculatorTemplate<unsigned int>;
REGISTER_CALCULATOR(StringToUintCalculator);

using StringToInt32Calculator = StringToIntCalculatorTemplate<int32_t>;
REGISTER_CALCULATOR(StringToInt32Calculator);

using StringToUint32Calculator = StringToIntCalculatorTemplate<uint32_t>;
REGISTER_CALCULATOR(StringToUint32Calculator);

using StringToInt64Calculator = StringToIntCalculatorTemplate<int64_t>;
REGISTER_CALCULATOR(StringToInt64Calculator);

using StringToUint64Calculator = StringToIntCalculatorTemplate<uint64_t>;
REGISTER_CALCULATOR(StringToUint64Calculator);

}  // namespace mediapipe
