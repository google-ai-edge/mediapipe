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

#include <vector>

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

TEST(TopKScoresCalculatorTest, TestNodeConfig) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "TopKScoresCalculator"
    input_stream: "SCORES:score_vector"
    output_stream: "TOP_K_INDEXES:top_k_indexes"
    output_stream: "TOP_K_SCORES:top_k_scores"
    options: {
      [mediapipe.TopKScoresCalculatorOptions.ext] {}
    }
  )"));

  auto status = runner.Run();
  ASSERT_TRUE(!status.ok());
  EXPECT_THAT(
      status.ToString(),
      testing::HasSubstr(
          "Must specify at least one of the top_k and threshold fields"));
}

TEST(TopKScoresCalculatorTest, TestTopKOnly) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "TopKScoresCalculator"
    input_stream: "SCORES:score_vector"
    output_stream: "TOP_K_INDEXES:top_k_indexes"
    output_stream: "TOP_K_SCORES:top_k_scores"
    options: {
      [mediapipe.TopKScoresCalculatorOptions.ext] { top_k: 2 }
    }
  )"));

  std::vector<float> score_vector{0.9, 0.2, 0.3, 1.0, 0.1};

  runner.MutableInputs()->Tag("SCORES").packets.push_back(
      MakePacket<std::vector<float>>(score_vector).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& indexes_outputs =
      runner.Outputs().Tag("TOP_K_INDEXES").packets;
  ASSERT_EQ(1, indexes_outputs.size());
  const auto& indexes = indexes_outputs[0].Get<std::vector<int>>();
  EXPECT_EQ(2, indexes.size());
  EXPECT_EQ(3, indexes[0]);
  EXPECT_EQ(0, indexes[1]);
  const std::vector<Packet>& scores_outputs =
      runner.Outputs().Tag("TOP_K_SCORES").packets;
  ASSERT_EQ(1, scores_outputs.size());
  const auto& scores = scores_outputs[0].Get<std::vector<float>>();
  EXPECT_EQ(2, scores.size());
  EXPECT_NEAR(1, scores[0], 1e-5);
  EXPECT_NEAR(0.9, scores[1], 1e-5);
}

TEST(TopKScoresCalculatorTest, TestThresholdOnly) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "TopKScoresCalculator"
    input_stream: "SCORES:score_vector"
    output_stream: "TOP_K_INDEXES:top_k_indexes"
    output_stream: "TOP_K_SCORES:top_k_scores"
    options: {
      [mediapipe.TopKScoresCalculatorOptions.ext] { threshold: 0.2 }
    }
  )"));

  std::vector<float> score_vector{0.9, 0.2, 0.3, 1.0, 0.1};

  runner.MutableInputs()->Tag("SCORES").packets.push_back(
      MakePacket<std::vector<float>>(score_vector).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& indexes_outputs =
      runner.Outputs().Tag("TOP_K_INDEXES").packets;
  ASSERT_EQ(1, indexes_outputs.size());
  const auto& indexes = indexes_outputs[0].Get<std::vector<int>>();
  EXPECT_EQ(4, indexes.size());
  EXPECT_EQ(3, indexes[0]);
  EXPECT_EQ(0, indexes[1]);
  EXPECT_EQ(2, indexes[2]);
  EXPECT_EQ(1, indexes[3]);
  const std::vector<Packet>& scores_outputs =
      runner.Outputs().Tag("TOP_K_SCORES").packets;
  ASSERT_EQ(1, scores_outputs.size());
  const auto& scores = scores_outputs[0].Get<std::vector<float>>();
  EXPECT_EQ(4, scores.size());
  EXPECT_NEAR(1.0, scores[0], 1e-5);
  EXPECT_NEAR(0.9, scores[1], 1e-5);
  EXPECT_NEAR(0.3, scores[2], 1e-5);
  EXPECT_NEAR(0.2, scores[3], 1e-5);
}

TEST(TopKScoresCalculatorTest, TestBothTopKAndThreshold) {
  CalculatorRunner runner(ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
    calculator: "TopKScoresCalculator"
    input_stream: "SCORES:score_vector"
    output_stream: "TOP_K_INDEXES:top_k_indexes"
    output_stream: "TOP_K_SCORES:top_k_scores"
    options: {
      [mediapipe.TopKScoresCalculatorOptions.ext] { top_k: 4 threshold: 0.3 }
    }
  )"));

  std::vector<float> score_vector{0.9, 0.2, 0.3, 1.0, 0.1};

  runner.MutableInputs()->Tag("SCORES").packets.push_back(
      MakePacket<std::vector<float>>(score_vector).At(Timestamp(0)));

  MP_ASSERT_OK(runner.Run());
  const std::vector<Packet>& indexes_outputs =
      runner.Outputs().Tag("TOP_K_INDEXES").packets;
  ASSERT_EQ(1, indexes_outputs.size());
  const auto& indexes = indexes_outputs[0].Get<std::vector<int>>();
  EXPECT_EQ(3, indexes.size());
  EXPECT_EQ(3, indexes[0]);
  EXPECT_EQ(0, indexes[1]);
  EXPECT_EQ(2, indexes[2]);
  const std::vector<Packet>& scores_outputs =
      runner.Outputs().Tag("TOP_K_SCORES").packets;
  ASSERT_EQ(1, scores_outputs.size());
  const auto& scores = scores_outputs[0].Get<std::vector<float>>();
  EXPECT_EQ(3, scores.size());
  EXPECT_NEAR(1.0, scores[0], 1e-5);
  EXPECT_NEAR(0.9, scores[1], 1e-5);
  EXPECT_NEAR(0.3, scores[2], 1e-5);
}

}  // namespace mediapipe
