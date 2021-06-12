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

#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

// Checks that the calculator fails if no input streams are provided.
TEST(InvariantMergeInputStreamsCalculator, NoInputStreamsMustFail) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeCalculator"
    output_stream: "merged_output"
  )pb"));
  // Expect calculator to fail.
  ASSERT_FALSE(runner.Run().ok());
}

// Checks that the calculator fails with an incorrect number of output streams.
TEST(InvariantMergeInputStreamsCalculator, ExpectExactlyOneOutputStream) {
  CalculatorRunner runner1(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeCalculator"
        input_stream: "input1"
        input_stream: "input2"
      )pb"));
  // Expect calculator to fail.
  EXPECT_FALSE(runner1.Run().ok());

  CalculatorRunner runner2(
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MergeCalculator"
        input_stream: "input1"
        input_stream: "input2"
        output_stream: "output1"
        output_stream: "output2"
      )pb"));
  // Expect calculator to fail.
  ASSERT_FALSE(runner2.Run().ok());
}

// Ensures two streams with differing types can be merged correctly.
TEST(MediaPipeDetectionToSoapboxDetectionCalculatorTest,
     TestMergingTwoStreams) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeCalculator"
    input_stream: "input1"
    input_stream: "input2"
    output_stream: "combined_output"
  )pb"));

  // input1: integers 10, 20, 30, occurring at times 10, 20, 30.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int(10)).At(Timestamp(10)));
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int(20)).At(Timestamp(20)));
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int(30)).At(Timestamp(30)));
  // input2: floats 5.5, 35.5 at times 5, 35.
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(new float(5.5)).At(Timestamp(5)));
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(new float(35.5)).At(Timestamp(35)));

  MP_ASSERT_OK(runner.Run());

  // Expected combined_output: 5.5, 10, 20, 30, 35.5 at times 5, 10, 20, 30, 35.
  const std::vector<Packet>& actual_output = runner.Outputs().Index(0).packets;
  ASSERT_EQ(actual_output.size(), 5);
  EXPECT_EQ(actual_output[0].Timestamp(), Timestamp(5));
  EXPECT_EQ(actual_output[0].Get<float>(), 5.5);

  EXPECT_EQ(actual_output[1].Timestamp(), Timestamp(10));
  EXPECT_EQ(actual_output[1].Get<int>(), 10);

  EXPECT_EQ(actual_output[2].Timestamp(), Timestamp(20));
  EXPECT_EQ(actual_output[2].Get<int>(), 20);

  EXPECT_EQ(actual_output[3].Timestamp(), Timestamp(30));
  EXPECT_EQ(actual_output[3].Get<int>(), 30);

  EXPECT_EQ(actual_output[4].Timestamp(), Timestamp(35));
  EXPECT_EQ(actual_output[4].Get<float>(), 35.5);
}

// Ensures three streams with differing types can be merged correctly.
TEST(MediaPipeDetectionToSoapboxDetectionCalculatorTest,
     TestMergingThreeStreams) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
    calculator: "MergeCalculator"
    input_stream: "input1"
    input_stream: "input2"
    input_stream: "input3"
    output_stream: "combined_output"
  )pb"));

  // input1: integer 30 occurring at time 30.
  runner.MutableInputs()->Index(0).packets.push_back(
      Adopt(new int(30)).At(Timestamp(30)));
  // input2: float 20.5 occurring at time 20.
  runner.MutableInputs()->Index(1).packets.push_back(
      Adopt(new float(20.5)).At(Timestamp(20)));
  // input3: char 'c' occurring at time 10.
  runner.MutableInputs()->Index(2).packets.push_back(
      Adopt(new char('c')).At(Timestamp(10)));

  MP_ASSERT_OK(runner.Run());

  // Expected combined_output: 'c', 20.5, 30 at times 10, 20, 30.
  const std::vector<Packet>& actual_output = runner.Outputs().Index(0).packets;
  ASSERT_EQ(actual_output.size(), 3);
  EXPECT_EQ(actual_output[0].Timestamp(), Timestamp(10));
  EXPECT_EQ(actual_output[0].Get<char>(), 'c');

  EXPECT_EQ(actual_output[1].Timestamp(), Timestamp(20));
  EXPECT_EQ(actual_output[1].Get<float>(), 20.5);

  EXPECT_EQ(actual_output[2].Timestamp(), Timestamp(30));
  EXPECT_EQ(actual_output[2].Get<int>(), 30);
}

}  // namespace
}  // namespace mediapipe
