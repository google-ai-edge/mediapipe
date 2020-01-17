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

#include "mediapipe/calculators/core/clip_vector_size_calculator.h"

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

typedef ClipVectorSizeCalculator<int> TestClipIntVectorSizeCalculator;
REGISTER_CALCULATOR(TestClipIntVectorSizeCalculator);

void AddInputVector(const std::vector<int>& input, int64 timestamp,
                    CalculatorRunner* runner) {
  runner->MutableInputs()->Index(0).packets.push_back(
      MakePacket<std::vector<int>>(input).At(Timestamp(timestamp)));
}

TEST(TestClipIntVectorSizeCalculatorTest, EmptyVectorInput) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "TestClipIntVectorSizeCalculator"
        input_stream: "input_vector"
        output_stream: "output_vector"
        options {
          [mediapipe.ClipVectorSizeCalculatorOptions.ext] { max_vec_size: 1 }
        }
      )");
  CalculatorRunner runner(node_config);

  std::vector<int> input = {};
  AddInputVector(input, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  EXPECT_TRUE(outputs[0].Get<std::vector<int>>().empty());
}

TEST(TestClipIntVectorSizeCalculatorTest, OneTimestamp) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "TestClipIntVectorSizeCalculator"
        input_stream: "input_vector"
        output_stream: "output_vector"
        options {
          [mediapipe.ClipVectorSizeCalculatorOptions.ext] { max_vec_size: 2 }
        }
      )");
  CalculatorRunner runner(node_config);

  std::vector<int> input = {0, 1, 2, 3};
  AddInputVector(input, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const std::vector<int>& output = outputs[0].Get<std::vector<int>>();
  EXPECT_EQ(2, output.size());
  std::vector<int> expected_vector = {0, 1};
  EXPECT_EQ(expected_vector, output);
}

TEST(TestClipIntVectorSizeCalculatorTest, TwoInputsAtTwoTimestamps) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "TestClipIntVectorSizeCalculator"
        input_stream: "input_vector"
        output_stream: "output_vector"
        options {
          [mediapipe.ClipVectorSizeCalculatorOptions.ext] { max_vec_size: 3 }
        }
      )");
  CalculatorRunner runner(node_config);

  {
    std::vector<int> input = {0, 1, 2, 3};
    AddInputVector(input, /*timestamp=*/1, &runner);
  }
  {
    std::vector<int> input = {2, 3, 4, 5};
    AddInputVector(input, /*timestamp=*/2, &runner);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(2, outputs.size());
  {
    EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
    const std::vector<int>& output = outputs[0].Get<std::vector<int>>();
    EXPECT_EQ(3, output.size());
    std::vector<int> expected_vector = {0, 1, 2};
    EXPECT_EQ(expected_vector, output);
  }
  {
    EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
    const std::vector<int>& output = outputs[1].Get<std::vector<int>>();
    EXPECT_EQ(3, output.size());
    std::vector<int> expected_vector = {2, 3, 4};
    EXPECT_EQ(expected_vector, output);
  }
}

typedef ClipVectorSizeCalculator<std::unique_ptr<int>>
    TestClipUniqueIntPtrVectorSizeCalculator;
REGISTER_CALCULATOR(TestClipUniqueIntPtrVectorSizeCalculator);

TEST(TestClipUniqueIntPtrVectorSizeCalculatorTest, ConsumeOneTimestamp) {
  /* Note: We don't use CalculatorRunner for this test because it keeps copies
   * of input packets, so packets sent to the graph don't have sole ownership.
   * The test needs to send packets that own the data.
   */
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_vector"
        node {
          calculator: "TestClipUniqueIntPtrVectorSizeCalculator"
          input_stream: "input_vector"
          output_stream: "output_vector"
          options {
            [mediapipe.ClipVectorSizeCalculatorOptions.ext] { max_vec_size: 3 }
          }
        }
      )");

  std::vector<Packet> outputs;
  tool::AddVectorSink("output_vector", &graph_config, &outputs);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));

  // input1 : {0, 1, 2, 3, 4, 5}
  auto input_vector = absl::make_unique<std::vector<std::unique_ptr<int>>>(6);
  for (int i = 0; i < 6; ++i) {
    input_vector->at(i) = absl::make_unique<int>(i);
  }

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "input_vector", Adopt(input_vector.release()).At(Timestamp(1))));

  MP_EXPECT_OK(graph.WaitUntilIdle());
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const std::vector<std::unique_ptr<int>>& result =
      outputs[0].Get<std::vector<std::unique_ptr<int>>>();
  EXPECT_EQ(3, result.size());
  for (int i = 0; i < 3; ++i) {
    const std::unique_ptr<int>& v = result[i];
    EXPECT_EQ(i, *v);
  }
}

TEST(TestClipIntVectorSizeCalculatorTest, SidePacket) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "TestClipIntVectorSizeCalculator"
        input_stream: "input_vector"
        input_side_packet: "max_vec_size"
        output_stream: "output_vector"
        options {
          [mediapipe.ClipVectorSizeCalculatorOptions.ext] { max_vec_size: 1 }
        }
      )");
  CalculatorRunner runner(node_config);
  // This should override the default of 1 set in the options.
  runner.MutableSidePackets()->Index(0) = Adopt(new int(2));
  std::vector<int> input = {0, 1, 2, 3};
  AddInputVector(input, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const std::vector<int>& output = outputs[0].Get<std::vector<int>>();
  EXPECT_EQ(2, output.size());
  std::vector<int> expected_vector = {0, 1};
  EXPECT_EQ(expected_vector, output);
}

}  // namespace mediapipe
