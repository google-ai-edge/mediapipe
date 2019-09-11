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

#include "mediapipe/calculators/core/concatenate_vector_calculator.h"

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {

typedef ConcatenateVectorCalculator<int> TestConcatenateIntVectorCalculator;
REGISTER_CALCULATOR(TestConcatenateIntVectorCalculator);

void AddInputVectors(const std::vector<std::vector<int>>& inputs,
                     int64 timestamp, CalculatorRunner* runner) {
  for (int i = 0; i < inputs.size(); ++i) {
    runner->MutableInputs()->Index(i).packets.push_back(
        MakePacket<std::vector<int>>(inputs[i]).At(Timestamp(timestamp)));
  }
}

TEST(TestConcatenateIntVectorCalculatorTest, EmptyVectorInputs) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<int>> inputs = {{}, {}, {}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].Get<std::vector<int>>().empty());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
}

TEST(TestConcatenateIntVectorCalculatorTest, OneTimestamp) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<int>> inputs = {{1, 2, 3}, {4}, {5, 6}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
}

TEST(TestConcatenateIntVectorCalculatorTest, TwoInputsAtTwoTimestamps) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);
  {
    std::vector<std::vector<int>> inputs = {{1, 2, 3}, {4}, {5, 6}};
    AddInputVectors(inputs, /*timestamp=*/1, &runner);
  }
  {
    std::vector<std::vector<int>> inputs = {{0, 2}, {1}, {3, 5}};
    AddInputVectors(inputs, /*timestamp=*/2, &runner);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(2, outputs.size());
  {
    EXPECT_EQ(6, outputs[0].Get<std::vector<int>>().size());
    EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
    std::vector<int> expected_vector = {1, 2, 3, 4, 5, 6};
    EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
  }
  {
    EXPECT_EQ(5, outputs[1].Get<std::vector<int>>().size());
    EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
    std::vector<int> expected_vector = {0, 2, 1, 3, 5};
    EXPECT_EQ(expected_vector, outputs[1].Get<std::vector<int>>());
  }
}

TEST(TestConcatenateIntVectorCalculatorTest, OneEmptyStreamStillOutput) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<int>> inputs = {{1, 2, 3}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2, 3};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
}

TEST(TestConcatenateIntVectorCalculatorTest, OneEmptyStreamNoOutput) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/
                          "[mediapipe.ConcatenateVectorCalculatorOptions.ext]: "
                          "{only_emit_if_all_present: true}",
                          /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<int>> inputs = {{1, 2, 3}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(0, outputs.size());
}

void AddInputVectors(const std::vector<std::vector<float>>& inputs,
                     int64 timestamp, CalculatorRunner* runner) {
  for (int i = 0; i < inputs.size(); ++i) {
    runner->MutableInputs()->Index(i).packets.push_back(
        MakePacket<std::vector<float>>(inputs[i]).At(Timestamp(timestamp)));
  }
}

TEST(ConcatenateFloatVectorCalculatorTest, EmptyVectorInputs) {
  CalculatorRunner runner("ConcatenateFloatVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<float>> inputs = {{}, {}, {}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].Get<std::vector<float>>().empty());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
}

TEST(ConcatenateFloatVectorCalculatorTest, OneTimestamp) {
  CalculatorRunner runner("ConcatenateFloatVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<float>> inputs = {
      {1.0f, 2.0f, 3.0f}, {4.0f}, {5.0f, 6.0f}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<float> expected_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<float>>());
}

TEST(ConcatenateFloatVectorCalculatorTest, TwoInputsAtTwoTimestamps) {
  CalculatorRunner runner("ConcatenateFloatVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);
  {
    std::vector<std::vector<float>> inputs = {
        {1.0f, 2.0f, 3.0f}, {4.0f}, {5.0f, 6.0f}};
    AddInputVectors(inputs, /*timestamp=*/1, &runner);
  }
  {
    std::vector<std::vector<float>> inputs = {
        {0.0f, 2.0f}, {1.0f}, {3.0f, 5.0f}};
    AddInputVectors(inputs, /*timestamp=*/2, &runner);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(2, outputs.size());
  {
    EXPECT_EQ(6, outputs[0].Get<std::vector<float>>().size());
    EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
    std::vector<float> expected_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<float>>());
  }
  {
    EXPECT_EQ(5, outputs[1].Get<std::vector<float>>().size());
    EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
    std::vector<float> expected_vector = {0.0f, 2.0f, 1.0f, 3.0f, 5.0f};
    EXPECT_EQ(expected_vector, outputs[1].Get<std::vector<float>>());
  }
}

TEST(ConcatenateFloatVectorCalculatorTest, OneEmptyStreamStillOutput) {
  CalculatorRunner runner("ConcatenateFloatVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<float>> inputs = {{1.0f, 2.0f, 3.0f}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<float> expected_vector = {1.0f, 2.0f, 3.0f};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<float>>());
}

TEST(ConcatenateFloatVectorCalculatorTest, OneEmptyStreamNoOutput) {
  CalculatorRunner runner("ConcatenateFloatVectorCalculator",
                          /*options_string=*/
                          "[mediapipe.ConcatenateVectorCalculatorOptions.ext]: "
                          "{only_emit_if_all_present: true}",
                          /*num_inputs=*/2,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<std::vector<float>> inputs = {{1.0f, 2.0f, 3.0f}};
  AddInputVectors(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(0, outputs.size());
}

}  // namespace mediapipe
