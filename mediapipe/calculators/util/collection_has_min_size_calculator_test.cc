// Copyright 2020 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/collection_has_min_size_calculator.h"

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

constexpr char kIterableTag[] = "ITERABLE";

typedef CollectionHasMinSizeCalculator<std::vector<int>>
    TestIntCollectionHasMinSizeCalculator;
REGISTER_CALCULATOR(TestIntCollectionHasMinSizeCalculator);

void AddInputVector(const std::vector<int>& input, int64_t timestamp,
                    CalculatorRunner* runner) {
  runner->MutableInputs()
      ->Tag(kIterableTag)
      .packets.push_back(
          MakePacket<std::vector<int>>(input).At(Timestamp(timestamp)));
}

TEST(TestIntCollectionHasMinSizeCalculator, DoesHaveMinSize) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TestIntCollectionHasMinSizeCalculator"
        input_stream: "ITERABLE:input_vector"
        output_stream: "output_vector"
        options {
          [mediapipe.CollectionHasMinSizeCalculatorOptions.ext] { min_size: 2 }
        }
      )pb");
  CalculatorRunner runner(node_config);
  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;

  AddInputVector({1, 2}, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  EXPECT_TRUE(outputs[0].Get<bool>());

  AddInputVector({1, 2, 3}, /*timestamp=*/2, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
  EXPECT_TRUE(outputs[1].Get<bool>());
}

TEST(TestIntCollectionHasMinSizeCalculator,
     DoesHaveMinSize_MinSizeAsSidePacket) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TestIntCollectionHasMinSizeCalculator"
        input_stream: "ITERABLE:input_vector"
        input_side_packet: "min_size"
        output_stream: "output_vector"
      )pb");
  CalculatorRunner runner(node_config);
  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;

  runner.MutableSidePackets()->Index(0) = MakePacket<int>(2);

  AddInputVector({1, 2}, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  EXPECT_TRUE(outputs[0].Get<bool>());

  AddInputVector({1, 2, 3}, /*timestamp=*/2, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
  EXPECT_TRUE(outputs[1].Get<bool>());
}

TEST(TestIntCollectionHasMinSizeCalculator, DoesNotHaveMinSize) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TestIntCollectionHasMinSizeCalculator"
        input_stream: "ITERABLE:input_vector"
        output_stream: "output_vector"
        options {
          [mediapipe.CollectionHasMinSizeCalculatorOptions.ext] { min_size: 3 }
        }
      )pb");
  CalculatorRunner runner(node_config);
  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;

  AddInputVector({1}, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  EXPECT_FALSE(outputs[0].Get<bool>());

  AddInputVector({1, 2}, /*timestamp=*/2, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
  EXPECT_FALSE(outputs[1].Get<bool>());
}

TEST(TestIntCollectionHasMinSizeCalculator,
     DoesNotHaveMinSize_MinSizeAsSidePacket) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "TestIntCollectionHasMinSizeCalculator"
        input_stream: "ITERABLE:input_vector"
        input_side_packet: "min_size"
        output_stream: "output_vector"
      )pb");
  CalculatorRunner runner(node_config);
  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;

  runner.MutableSidePackets()->Index(0) = MakePacket<int>(3);

  AddInputVector({1}, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  EXPECT_FALSE(outputs[0].Get<bool>());

  AddInputVector({1, 2}, /*timestamp=*/2, &runner);
  MP_ASSERT_OK(runner.Run());

  EXPECT_EQ(2, outputs.size());
  EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
  EXPECT_FALSE(outputs[1].Get<bool>());
}

}  // namespace mediapipe
