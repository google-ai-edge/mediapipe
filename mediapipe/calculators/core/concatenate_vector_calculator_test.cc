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

void AddInputVector(int index, const std::vector<int>& input, int64 timestamp,
                    CalculatorRunner* runner) {
  runner->MutableInputs()->Index(index).packets.push_back(
      MakePacket<std::vector<int>>(input).At(Timestamp(timestamp)));
}

void AddInputVectors(const std::vector<std::vector<int>>& inputs,
                     int64 timestamp, CalculatorRunner* runner) {
  for (int i = 0; i < inputs.size(); ++i) {
    AddInputVector(i, inputs[i], timestamp, runner);
  }
}

void AddInputItem(int index, int input, int64 timestamp,
                  CalculatorRunner* runner) {
  runner->MutableInputs()->Index(index).packets.push_back(
      MakePacket<int>(input).At(Timestamp(timestamp)));
}

void AddInputItems(const std::vector<int>& inputs, int64 timestamp,
                   CalculatorRunner* runner) {
  for (int i = 0; i < inputs.size(); ++i) {
    AddInputItem(i, inputs[i], timestamp, runner);
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

TEST(TestConcatenateIntVectorCalculatorTest, ItemsOneTimestamp) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<int> inputs = {1, 2, 3};
  AddInputItems(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2, 3};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
}

TEST(TestConcatenateIntVectorCalculatorTest, ItemsTwoInputsAtTwoTimestamps) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  {
    std::vector<int> inputs = {1, 2, 3};
    AddInputItems(inputs, /*timestamp=*/1, &runner);
  }
  {
    std::vector<int> inputs = {4, 5, 6};
    AddInputItems(inputs, /*timestamp=*/2, &runner);
  }
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(2, outputs.size());
  {
    EXPECT_EQ(3, outputs[0].Get<std::vector<int>>().size());
    EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
    std::vector<int> expected_vector = {1, 2, 3};
    EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
  }
  {
    EXPECT_EQ(3, outputs[1].Get<std::vector<int>>().size());
    EXPECT_EQ(Timestamp(2), outputs[1].Timestamp());
    std::vector<int> expected_vector = {4, 5, 6};
    EXPECT_EQ(expected_vector, outputs[1].Get<std::vector<int>>());
  }
}

TEST(TestConcatenateIntVectorCalculatorTest, ItemsOneEmptyStreamStillOutput) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  // No third input item.
  std::vector<int> inputs = {1, 2};
  AddInputItems(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
}

TEST(TestConcatenateIntVectorCalculatorTest, ItemsOneEmptyStreamNoOutput) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/
                          "[mediapipe.ConcatenateVectorCalculatorOptions.ext]: "
                          "{only_emit_if_all_present: true}",
                          /*num_inputs=*/3,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  // No third input item.
  std::vector<int> inputs = {1, 2};
  AddInputItems(inputs, /*timestamp=*/1, &runner);
  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(0, outputs.size());
}

TEST(TestConcatenateIntVectorCalculatorTest, MixedVectorsAndItems) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/4,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  std::vector<int> vector_0 = {1, 2};
  std::vector<int> vector_1 = {3, 4, 5};
  int item_0 = 6;
  int item_1 = 7;

  AddInputVector(/*index*/ 0, vector_0, /*timestamp=*/1, &runner);
  AddInputVector(/*index*/ 1, vector_1, /*timestamp=*/1, &runner);
  AddInputItem(/*index*/ 2, item_0, /*timestamp=*/1, &runner);
  AddInputItem(/*index*/ 3, item_1, /*timestamp=*/1, &runner);

  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
}

TEST(TestConcatenateIntVectorCalculatorTest, MixedVectorsAndItemsAnother) {
  CalculatorRunner runner("TestConcatenateIntVectorCalculator",
                          /*options_string=*/"", /*num_inputs=*/4,
                          /*num_outputs=*/1, /*num_side_packets=*/0);

  int item_0 = 1;
  std::vector<int> vector_0 = {2, 3};
  std::vector<int> vector_1 = {4, 5, 6};
  int item_1 = 7;

  AddInputItem(/*index*/ 0, item_0, /*timestamp=*/1, &runner);
  AddInputVector(/*index*/ 1, vector_0, /*timestamp=*/1, &runner);
  AddInputVector(/*index*/ 2, vector_1, /*timestamp=*/1, &runner);
  AddInputItem(/*index*/ 3, item_1, /*timestamp=*/1, &runner);

  MP_ASSERT_OK(runner.Run());

  const std::vector<Packet>& outputs = runner.Outputs().Index(0).packets;
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  std::vector<int> expected_vector = {1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(expected_vector, outputs[0].Get<std::vector<int>>());
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

typedef ConcatenateVectorCalculator<std::unique_ptr<int>>
    TestConcatenateUniqueIntPtrCalculator;
REGISTER_CALCULATOR(TestConcatenateUniqueIntPtrCalculator);

TEST(TestConcatenateUniqueIntVectorCalculatorTest, ConsumeOneTimestamp) {
  /* Note: We don't use CalculatorRunner for this test because it keeps copies
   * of input packets, so packets sent to the graph don't have sole ownership.
   * The test needs to send packets that own the data.
   */
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in_1"
        input_stream: "in_2"
        input_stream: "in_3"
        node {
          calculator: "TestConcatenateUniqueIntPtrCalculator"
          input_stream: "in_1"
          input_stream: "in_2"
          input_stream: "in_3"
          output_stream: "out"
        }
      )");

  std::vector<Packet> outputs;
  tool::AddVectorSink("out", &graph_config, &outputs);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));

  // input1 : {0, 1, 2}
  std::unique_ptr<std::vector<std::unique_ptr<int>>> input_1 =
      absl::make_unique<std::vector<std::unique_ptr<int>>>(3);
  for (int i = 0; i < 3; ++i) {
    input_1->at(i) = absl::make_unique<int>(i);
  }
  // input2: {3}
  std::unique_ptr<std::vector<std::unique_ptr<int>>> input_2 =
      absl::make_unique<std::vector<std::unique_ptr<int>>>(1);
  input_2->at(0) = absl::make_unique<int>(3);
  // input3: {4, 5}
  std::unique_ptr<std::vector<std::unique_ptr<int>>> input_3 =
      absl::make_unique<std::vector<std::unique_ptr<int>>>(2);
  input_3->at(0) = absl::make_unique<int>(4);
  input_3->at(1) = absl::make_unique<int>(5);

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_1", Adopt(input_1.release()).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_2", Adopt(input_2.release()).At(Timestamp(1))));
  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_3", Adopt(input_3.release()).At(Timestamp(1))));

  MP_EXPECT_OK(graph.WaitUntilIdle());
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());

  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(Timestamp(1), outputs[0].Timestamp());
  const std::vector<std::unique_ptr<int>>& result =
      outputs[0].Get<std::vector<std::unique_ptr<int>>>();
  EXPECT_EQ(6, result.size());
  for (int i = 0; i < 6; ++i) {
    const std::unique_ptr<int>& v = result[i];
    EXPECT_EQ(i, *v);
  }
}

TEST(TestConcatenateUniqueIntVectorCalculatorTest, OneEmptyStreamStillOutput) {
  /* Note: We don't use CalculatorRunner for this test because it keeps copies
   * of input packets, so packets sent to the graph don't have sole ownership.
   * The test needs to send packets that own the data.
   */
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in_1"
        input_stream: "in_2"
        node {
          calculator: "TestConcatenateUniqueIntPtrCalculator"
          input_stream: "in_1"
          input_stream: "in_2"
          output_stream: "out"
        }
      )");

  std::vector<Packet> outputs;
  tool::AddVectorSink("out", &graph_config, &outputs);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));

  // input1 : {0, 1, 2}
  std::unique_ptr<std::vector<std::unique_ptr<int>>> input_1 =
      absl::make_unique<std::vector<std::unique_ptr<int>>>(3);
  for (int i = 0; i < 3; ++i) {
    input_1->at(i) = absl::make_unique<int>(i);
  }

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_1", Adopt(input_1.release()).At(Timestamp(1))));

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

TEST(TestConcatenateUniqueIntVectorCalculatorTest, OneEmptyStreamNoOutput) {
  /* Note: We don't use CalculatorRunner for this test because it keeps copies
   * of input packets, so packets sent to the graph don't have sole ownership.
   * The test needs to send packets that own the data.
   */
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in_1"
        input_stream: "in_2"
        node {
          calculator: "TestConcatenateUniqueIntPtrCalculator"
          input_stream: "in_1"
          input_stream: "in_2"
          output_stream: "out"
          options {
            [mediapipe.ConcatenateVectorCalculatorOptions.ext] {
              only_emit_if_all_present: true
            }
          }
        }
      )");

  std::vector<Packet> outputs;
  tool::AddVectorSink("out", &graph_config, &outputs);

  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config));
  MP_EXPECT_OK(graph.StartRun({}));

  // input1 : {0, 1, 2}
  std::unique_ptr<std::vector<std::unique_ptr<int>>> input_1 =
      absl::make_unique<std::vector<std::unique_ptr<int>>>(3);
  for (int i = 0; i < 3; ++i) {
    input_1->at(i) = absl::make_unique<int>(i);
  }

  MP_EXPECT_OK(graph.AddPacketToInputStream(
      "in_1", Adopt(input_1.release()).At(Timestamp(1))));

  MP_EXPECT_OK(graph.WaitUntilIdle());
  MP_EXPECT_OK(graph.CloseAllPacketSources());
  MP_EXPECT_OK(graph.WaitUntilDone());

  EXPECT_EQ(0, outputs.size());
}

}  // namespace mediapipe
