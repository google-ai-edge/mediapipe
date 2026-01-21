// Copyright 2025 The MediaPipe Authors.
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

#include "absl/status/status.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using ::testing::TestWithParam;

struct SourceCalculatorParam {
  std::string test_name;
  int initial_value;
  int max_count;
  int error_count;
  int batch_size;
  int increment;
  std::vector<int> expected_values = {};
};

using NegativeValueParamErrorTest = TestWithParam<SourceCalculatorParam>;
using VerifyOutputWithMaxCountSetTest = TestWithParam<SourceCalculatorParam>;
using VerifyOutputWithErrorCountSetTest = TestWithParam<SourceCalculatorParam>;

TEST(CountingSourceCalculatorTest, ErrorCountAndMaxCountNotSet) {
  // Either max_count or error_count should be set.
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    node {
      calculator: "CountingSourceCalculator"
      input_side_packet: "INITIAL_VALUE:initial_value"
      output_stream: "output_stream"
    }
  )pb");
  CalculatorGraph graph;
  EXPECT_FALSE(graph.Initialize(graph_config).ok());
}

TEST(CountingSourceCalculatorTest, ExpectedErrorAfterSetErrorOnOpen) {
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    node {
      calculator: "CountingSourceCalculator"
      input_side_packet: "ERROR_COUNT:error_count"
      input_side_packet: "ERROR_ON_OPEN:error_on_open"
      output_stream: "output_stream"
    }
  )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  EXPECT_FALSE(graph
                   .StartRun({
                       {"error_on_open", MakePacket<bool>(true)},
                   })
                   .ok());
}

TEST_P(NegativeValueParamErrorTest, NegativeValueParamError) {
  const SourceCalculatorParam& test_params = GetParam();
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    node {
      calculator: "CountingSourceCalculator"
      input_side_packet: "ERROR_COUNT:error_count"
      input_side_packet: "MAX_COUNT:max_count"
      input_side_packet: "BATCH_SIZE:batch_size"
      input_side_packet: "INCREMENT:increment"
      output_stream: "output_stream"
    }
  )pb");

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));
  MP_ASSERT_OK(graph.StartRun({
      {"error_count",
       MakePacket<int>(test_params.error_count).At(Timestamp(0))},
      {"max_count", MakePacket<int>(test_params.max_count).At(Timestamp(0))},
      {"batch_size", MakePacket<int>(test_params.batch_size).At(Timestamp(0))},
      {"increment", MakePacket<int>(test_params.increment).At(Timestamp(0))},
  }));
  EXPECT_THAT(graph.WaitUntilIdle(),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("CalculatorGraph::Run() failed")));
}

INSTANTIATE_TEST_SUITE_P(
    NegativeValueParamError, NegativeValueParamErrorTest,
    testing::ValuesIn<SourceCalculatorParam>({
        {.test_name = "NegMaxCount",
         .initial_value = 0,
         .max_count = -1,
         .error_count = 0,
         .batch_size = 0,
         .increment = 0},
        {.test_name = "NegErrorCount",
         .initial_value = 0,
         .max_count = 0,
         .error_count = -1,
         .batch_size = 0,
         .increment = 0},
        {.test_name = "NegBatchSize",
         .initial_value = 0,
         .max_count = 0,
         .error_count = 0,
         .batch_size = -1,
         .increment = 0},
        {.test_name = "NegIncrement",
         .initial_value = 0,
         .max_count = 2,
         .error_count = 2,
         .batch_size = 1,
         .increment = -1},
    }),
    [](const testing::TestParamInfo<SourceCalculatorParam>& info) {
      return info.param.test_name;
    });

TEST_P(VerifyOutputWithMaxCountSetTest, VerifyOutputWithMaxCountSet) {
  const SourceCalculatorParam& test_params = GetParam();
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    output_stream: "output_stream"
    node {
      calculator: "CountingSourceCalculator"
      input_side_packet: "MAX_COUNT:max_count"
      input_side_packet: "BATCH_SIZE:batch_size"
      input_side_packet: "INITIAL_VALUE:initial_value"
      input_side_packet: "INCREMENT:increment"
      output_stream: "output_stream"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_stream", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));

  MP_ASSERT_OK(graph.StartRun({
      {"max_count", MakePacket<int>(test_params.max_count).At(Timestamp(0))},
      {"batch_size", MakePacket<int>(test_params.batch_size).At(Timestamp(0))},
      {"initial_value",
       MakePacket<int>(test_params.initial_value).At(Timestamp(0))},
      {"increment", MakePacket<int>(test_params.increment).At(Timestamp(0))},
  }));

  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(output_packets.size(),
            test_params.max_count * test_params.batch_size);
  for (int i = 0; i < test_params.expected_values.size(); ++i) {
    EXPECT_EQ(output_packets[i].Get<int>(), test_params.expected_values[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    VerifyOutputWithMaxCountSet, VerifyOutputWithMaxCountSetTest,
    testing::ValuesIn<SourceCalculatorParam>({
        {.test_name = "MaxCountTestBasicWithBatchSize1Increment1",
         .initial_value = 0,
         .max_count = 3,
         .batch_size = 1,
         .increment = 1,
         .expected_values = {0, 1, 2}},
        {.test_name = "MaxCountTestWithNegativeInitialValue",
         .initial_value = -2,
         .max_count = 5,
         .batch_size = 1,
         .increment = 1,
         .expected_values = {-2, -1, 0, 1, 2}},
        {.test_name = "MaxCountTestWithIncrement2",
         .initial_value = 0,
         .max_count = 3,
         .batch_size = 1,
         .increment = 2,
         .expected_values = {0, 2, 4}},
        {.test_name = "MaxCountTestWithBatchSize2Increment2",
         .initial_value = 0,
         .max_count = 3,
         .batch_size = 2,
         .increment = 2,
         .expected_values = {0, 2, 4, 6, 8, 10}},
    }),
    [](const testing::TestParamInfo<SourceCalculatorParam>& info) {
      return info.param.test_name;
    });

TEST_P(VerifyOutputWithErrorCountSetTest, VerifyOutputWithErrorCountSet) {
  const SourceCalculatorParam& test_params = GetParam();
  auto graph_config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    node {
      calculator: "CountingSourceCalculator"
      input_side_packet: "ERROR_COUNT:error_count"
      input_side_packet: "MAX_COUNT:max_count"
      input_side_packet: "BATCH_SIZE:batch_size"
      input_side_packet: "INITIAL_VALUE:initial_value"
      input_side_packet: "INCREMENT:increment"
      output_stream: "output_stream"
    }
  )pb");

  std::vector<Packet> output_packets;
  tool::AddVectorSink("output_stream", &graph_config, &output_packets);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(graph_config));

  MP_ASSERT_OK(graph.StartRun({
      {"error_count",
       MakePacket<int>(test_params.error_count).At(Timestamp(0))},
      {"max_count", MakePacket<int>(test_params.max_count).At(Timestamp(0))},
      {"batch_size", MakePacket<int>(test_params.batch_size).At(Timestamp(0))},
      {"initial_value",
       MakePacket<int>(test_params.initial_value).At(Timestamp(0))},
      {"increment", MakePacket<int>(test_params.increment).At(Timestamp(0))},
  }));

  if (test_params.max_count >= test_params.error_count) {
    EXPECT_FALSE(graph.WaitUntilDone().ok());
    EXPECT_EQ(test_params.expected_values.size(), 0);
    return;
  }

  MP_ASSERT_OK(graph.WaitUntilIdle());
  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());

  EXPECT_EQ(output_packets.size(),
            test_params.max_count * test_params.batch_size);
  for (int i = 0; i < test_params.expected_values.size(); ++i) {
    EXPECT_EQ(output_packets[i].Get<int>(), test_params.expected_values[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    VerifyOutputWithErrorCountSet, VerifyOutputWithErrorCountSetTest,
    testing::ValuesIn<SourceCalculatorParam>({
        {.test_name = "ErrorCountLargerThanMaxCountWithBatchSize1",
         .initial_value = 0,
         .max_count = 3,
         .error_count = 5,
         .batch_size = 1,
         .increment = 1,
         .expected_values = {0, 1, 2}},
        {.test_name = "ErrorCountLargerThanMaxCountWithBatchSize2",
         .initial_value = 0,
         .max_count = 3,
         .error_count = 5,
         .batch_size = 2,
         .increment = 2,
         .expected_values = {0, 2, 4, 6, 8, 10}},
        {.test_name = "ErrorCountSmallerThanMaxCountProduceNoOutput0",
         .initial_value = 0,
         .max_count = 5,
         .error_count = 2,
         .batch_size = 1,
         .increment = 1,
         .expected_values = {}},
        {.test_name = "ErrorCountSmallerThanMaxCountProduceNoOutput1",
         .initial_value = 0,
         .max_count = 3,
         .error_count = 2,
         .batch_size = 2,
         .increment = 2,
         .expected_values = {}},
    }),
    [](const testing::TestParamInfo<SourceCalculatorParam>& info) {
      return info.param.test_name;
    });

}  // namespace mediapipe
