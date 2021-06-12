// Copyright 2021 The MediaPipe Authors.
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
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/tool/validate_type.h"

namespace mediapipe {

class NonZeroCalculatorTest : public ::testing::Test {
 protected:
  NonZeroCalculatorTest()
      : runner_(
            R"pb(
              calculator: "NonZeroCalculator"
              input_stream: "INPUT:input"
              output_stream: "OUTPUT:output"
              output_stream: "OUTPUT_BOOL:output_bool"
            )pb") {}

  void SetInput(const std::vector<int>& inputs) {
    int timestamp = 0;
    for (const auto input : inputs) {
      runner_.MutableInputs()
          ->Get("INPUT", 0)
          .packets.push_back(MakePacket<int>(input).At(Timestamp(timestamp++)));
    }
  }

  std::vector<int> GetOutput() {
    std::vector<int> result;
    for (const auto output : runner_.Outputs().Get("OUTPUT", 0).packets) {
      result.push_back(output.Get<int>());
    }
    return result;
  }

  std::vector<bool> GetOutputBool() {
    std::vector<bool> result;
    for (const auto output : runner_.Outputs().Get("OUTPUT_BOOL", 0).packets) {
      result.push_back(output.Get<bool>());
    }
    return result;
  }

  CalculatorRunner runner_;
};

TEST_F(NonZeroCalculatorTest, ProducesZeroOutputForZeroInput) {
  SetInput({0});

  MP_ASSERT_OK(runner_.Run());

  EXPECT_THAT(GetOutput(), ::testing::ElementsAre(0));
  EXPECT_THAT(GetOutputBool(), ::testing::ElementsAre(false));
}

TEST_F(NonZeroCalculatorTest, ProducesNonZeroOutputForNonZeroInput) {
  SetInput({1, 2, 3, -4, 5});

  MP_ASSERT_OK(runner_.Run());

  EXPECT_THAT(GetOutput(), ::testing::ElementsAre(1, 1, 1, 1, 1));
  EXPECT_THAT(GetOutputBool(),
              ::testing::ElementsAre(true, true, true, true, true));
}

TEST_F(NonZeroCalculatorTest, SwitchesBetweenNonZeroAndZeroOutput) {
  SetInput({1, 0, 3, 0, 5});
  MP_ASSERT_OK(runner_.Run());
  EXPECT_THAT(GetOutput(), ::testing::ElementsAre(1, 0, 1, 0, 1));
  EXPECT_THAT(GetOutputBool(),
              ::testing::ElementsAre(true, false, true, false, true));
}

}  // namespace mediapipe
