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

#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/core/begin_loop_calculator.h"
#include "mediapipe/calculators/core/end_loop_calculator.h"
#include "mediapipe/framework/calculator_contract.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {
namespace {

typedef BeginLoopCalculator<std::vector<int>> BeginLoopIntegerCalculator;
REGISTER_CALCULATOR(BeginLoopIntegerCalculator);

class IncrementCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const int& input_int = cc->Inputs().Index(0).Get<int>();
    auto output_int = absl::make_unique<int>(input_int + 1);
    cc->Outputs().Index(0).Add(output_int.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(IncrementCalculator);

typedef EndLoopCalculator<std::vector<int>> EndLoopIntegersCalculator;
REGISTER_CALCULATOR(EndLoopIntegersCalculator);

class BeginEndLoopCalculatorGraphTest : public ::testing::Test {
 protected:
  BeginEndLoopCalculatorGraphTest() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"(
          num_threads: 4
          input_stream: "ints"
          node {
            calculator: "BeginLoopIntegerCalculator"
            input_stream: "ITERABLE:ints"
            output_stream: "ITEM:int"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "IncrementCalculator"
            input_stream: "int"
            output_stream: "int_plus_one"
          }
          node {
            calculator: "EndLoopIntegersCalculator"
            input_stream: "ITEM:int_plus_one"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITERABLE:ints_plus_one"
          }
        )");
    tool::AddVectorSink("ints_plus_one", &graph_config_, &output_packets_);
  }

  CalculatorGraphConfig graph_config_;
  std::vector<Packet> output_packets_;
};

TEST_F(BeginEndLoopCalculatorGraphTest, SingleEmptyVector) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));
  auto input_vector = absl::make_unique<std::vector<int>>();
  Timestamp input_timestamp = Timestamp(0);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector.release()).At(input_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  ASSERT_EQ(0, output_packets_.size());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphTest, SingleNonEmptyVector) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));
  auto input_vector = absl::make_unique<std::vector<int>>();
  input_vector->emplace_back(0);
  input_vector->emplace_back(1);
  input_vector->emplace_back(2);
  Timestamp input_timestamp = Timestamp(0);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector.release()).At(input_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_EQ(1, output_packets_.size());
  EXPECT_EQ(input_timestamp, output_packets_[0].Timestamp());
  std::vector<int> expected_output_vector = {1, 2, 3};
  EXPECT_EQ(expected_output_vector, output_packets_[0].Get<std::vector<int>>());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphTest, MultipleVectors) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));

  auto input_vector0 = absl::make_unique<std::vector<int>>();
  input_vector0->emplace_back(0);
  input_vector0->emplace_back(1);
  Timestamp input_timestamp0 = Timestamp(0);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector0.release()).At(input_timestamp0)));

  auto input_vector1 = absl::make_unique<std::vector<int>>();
  Timestamp input_timestamp1 = Timestamp(1);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector1.release()).At(input_timestamp1)));

  auto input_vector2 = absl::make_unique<std::vector<int>>();
  input_vector2->emplace_back(2);
  input_vector2->emplace_back(3);
  Timestamp input_timestamp2 = Timestamp(2);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector2.release()).At(input_timestamp2)));

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(2, output_packets_.size());

  EXPECT_EQ(input_timestamp0, output_packets_[0].Timestamp());
  std::vector<int> expected_output_vector0 = {1, 2};
  EXPECT_EQ(expected_output_vector0,
            output_packets_[0].Get<std::vector<int>>());

  // At input_timestamp1, EndLoopCalc will forward timestamp bound as there are
  // no elements in vector to process.

  EXPECT_EQ(input_timestamp2, output_packets_[1].Timestamp());
  std::vector<int> expected_output_vector2 = {3, 4};
  EXPECT_EQ(expected_output_vector2,
            output_packets_[1].Get<std::vector<int>>());
}

class MultiplierCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Index(0).Set<int>();
    cc->Inputs().Index(1).Set<int>();
    cc->Outputs().Index(0).Set<int>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    const int& input_int = cc->Inputs().Index(0).Get<int>();
    const int& multiplier_int = cc->Inputs().Index(1).Get<int>();
    auto output_int = absl::make_unique<int>(input_int * multiplier_int);
    cc->Outputs().Index(0).Add(output_int.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
  }
};

REGISTER_CALCULATOR(MultiplierCalculator);

class BeginEndLoopCalculatorGraphWithClonedInputsTest : public ::testing::Test {
 protected:
  BeginEndLoopCalculatorGraphWithClonedInputsTest() {
    graph_config_ = ParseTextProtoOrDie<CalculatorGraphConfig>(
        R"(
          num_threads: 4
          input_stream: "ints"
          input_stream: "multiplier"
          node {
            calculator: "BeginLoopIntegerCalculator"
            input_stream: "ITERABLE:ints"
            input_stream: "CLONE:multiplier"
            output_stream: "ITEM:int_at_loop"
            output_stream: "CLONE:multiplier_cloned_at_loop"
            output_stream: "BATCH_END:timestamp"
          }
          node {
            calculator: "MultiplierCalculator"
            input_stream: "int_at_loop"
            input_stream: "multiplier_cloned_at_loop"
            output_stream: "multiplied_int_at_loop"
          }
          node {
            calculator: "EndLoopIntegersCalculator"
            input_stream: "ITEM:multiplied_int_at_loop"
            input_stream: "BATCH_END:timestamp"
            output_stream: "ITERABLE:multiplied_ints"
          }
        )");
    tool::AddVectorSink("multiplied_ints", &graph_config_, &output_packets_);
  }

  CalculatorGraphConfig graph_config_;
  std::vector<Packet> output_packets_;
};

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, SingleEmptyVector) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));
  auto input_vector = absl::make_unique<std::vector<int>>();
  Timestamp input_timestamp = Timestamp(42);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector.release()).At(input_timestamp)));
  auto multiplier = absl::make_unique<int>(2);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "multiplier", Adopt(multiplier.release()).At(input_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  // EndLoopCalc will forward the timestamp bound because there are no elements
  // in collection to output.
  ASSERT_EQ(0, output_packets_.size());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, SingleNonEmptyVector) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));
  auto input_vector = absl::make_unique<std::vector<int>>();
  input_vector->emplace_back(0);
  input_vector->emplace_back(1);
  input_vector->emplace_back(2);
  Timestamp input_timestamp = Timestamp(42);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector.release()).At(input_timestamp)));
  auto multiplier = absl::make_unique<int>(2);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "multiplier", Adopt(multiplier.release()).At(input_timestamp)));
  MP_ASSERT_OK(graph.WaitUntilIdle());

  ASSERT_EQ(1, output_packets_.size());
  EXPECT_EQ(input_timestamp, output_packets_[0].Timestamp());
  std::vector<int> expected_output_vector = {0, 2, 4};
  EXPECT_EQ(expected_output_vector, output_packets_[0].Get<std::vector<int>>());

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());
}

TEST_F(BeginEndLoopCalculatorGraphWithClonedInputsTest, MultipleVectors) {
  CalculatorGraph graph;
  MP_EXPECT_OK(graph.Initialize(graph_config_));
  MP_EXPECT_OK(graph.StartRun({}));

  auto input_vector0 = absl::make_unique<std::vector<int>>();
  input_vector0->emplace_back(0);
  input_vector0->emplace_back(1);
  Timestamp input_timestamp0 = Timestamp(42);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector0.release()).At(input_timestamp0)));
  auto multiplier0 = absl::make_unique<int>(2);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "multiplier", Adopt(multiplier0.release()).At(input_timestamp0)));

  auto input_vector1 = absl::make_unique<std::vector<int>>();
  Timestamp input_timestamp1 = Timestamp(43);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector1.release()).At(input_timestamp1)));
  auto multiplier1 = absl::make_unique<int>(2);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "multiplier", Adopt(multiplier1.release()).At(input_timestamp1)));

  auto input_vector2 = absl::make_unique<std::vector<int>>();
  input_vector2->emplace_back(2);
  input_vector2->emplace_back(3);
  Timestamp input_timestamp2 = Timestamp(44);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "ints", Adopt(input_vector2.release()).At(input_timestamp2)));
  auto multiplier2 = absl::make_unique<int>(3);
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "multiplier", Adopt(multiplier2.release()).At(input_timestamp2)));

  MP_ASSERT_OK(graph.CloseAllPacketSources());
  MP_ASSERT_OK(graph.WaitUntilDone());

  ASSERT_EQ(2, output_packets_.size());

  EXPECT_EQ(input_timestamp0, output_packets_[0].Timestamp());
  std::vector<int> expected_output_vector0 = {0, 2};
  EXPECT_EQ(expected_output_vector0,
            output_packets_[0].Get<std::vector<int>>());

  // At input_timestamp1, EndLoopCalc will forward timestamp bound as there are
  // no elements in vector to process.

  EXPECT_EQ(input_timestamp2, output_packets_[1].Timestamp());
  std::vector<int> expected_output_vector2 = {6, 9};
  EXPECT_EQ(expected_output_vector2,
            output_packets_[1].Get<std::vector<int>>());
}

}  // namespace
}  // namespace mediapipe
