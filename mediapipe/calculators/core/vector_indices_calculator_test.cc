// Copyright 2022 The MediaPipe Authors.
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

#include "mediapipe/calculators/core/vector_indices_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

template <typename T>
void AddInputVector(CalculatorRunner& runner, const std::vector<T>& inputs,
                    int timestamp) {
  runner.MutableInputs()->Tag("VECTOR").packets.push_back(
      MakePacket<std::vector<T>>(inputs).At(Timestamp(timestamp)));
}

template <typename T>
struct TestParams {
  const std::string test_name;
  const std::vector<T> inputs;
  const int timestamp;
  const std::vector<int> expected_indices;
};

class IntVectorIndicesCalculatorTest
    : public testing::TestWithParam<TestParams<int>> {};

TEST_P(IntVectorIndicesCalculatorTest, Succeeds) {
  CalculatorRunner runner = CalculatorRunner(R"(
    calculator: "IntVectorIndicesCalculator"
    input_stream: "VECTOR:vector_stream"
    output_stream: "INDICES:indices_stream"
  )");
  const std::vector<int>& inputs = GetParam().inputs;
  std::vector<int> expected_indices(inputs.size());
  AddInputVector(runner, inputs, GetParam().timestamp);
  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& outputs = runner.Outputs().Tag("INDICES").packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_THAT(outputs[0].Get<std::vector<int>>(),
              testing::ElementsAreArray(GetParam().expected_indices));
}

INSTANTIATE_TEST_SUITE_P(
    IntVectorIndicesCalculatorTest, IntVectorIndicesCalculatorTest,
    Values(TestParams<int>{
               /* test_name= */ "IntVectorIndices",
               /* inputs= */ {1, 2, 3},
               /* timestamp= */ 1,
               /* expected_indices= */ {0, 1, 2},
           },
           TestParams<int>{
               /* test_name= */ "EmptyVector",
               /* inputs= */ {},
               /* timestamp= */ 1,
               /* expected_indices= */ {},
           }),
    [](const TestParamInfo<IntVectorIndicesCalculatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace mediapipe
