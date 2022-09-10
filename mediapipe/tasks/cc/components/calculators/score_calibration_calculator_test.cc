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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

using ::mediapipe::ParseTextProtoOrDie;
using ::testing::HasSubstr;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

// Builds the graph and feeds inputs.
void BuildGraph(CalculatorRunner* runner, std::vector<float> scores,
                std::optional<std::vector<int>> indices = std::nullopt) {
  auto scores_tensors = std::make_unique<std::vector<Tensor>>();
  scores_tensors->emplace_back(
      Tensor::ElementType::kFloat32,
      Tensor::Shape{1, static_cast<int>(scores.size())});
  auto scores_view = scores_tensors->back().GetCpuWriteView();
  float* scores_buffer = scores_view.buffer<float>();
  ASSERT_NE(scores_buffer, nullptr);
  for (int i = 0; i < scores.size(); ++i) {
    scores_buffer[i] = scores[i];
  }
  auto& input_scores_packets = runner->MutableInputs()->Tag("SCORES").packets;
  input_scores_packets.push_back(
      mediapipe::Adopt(scores_tensors.release()).At(mediapipe::Timestamp(0)));

  if (indices.has_value()) {
    auto indices_tensors = std::make_unique<std::vector<Tensor>>();
    indices_tensors->emplace_back(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, static_cast<int>(indices->size())});
    auto indices_view = indices_tensors->back().GetCpuWriteView();
    float* indices_buffer = indices_view.buffer<float>();
    ASSERT_NE(indices_buffer, nullptr);
    for (int i = 0; i < indices->size(); ++i) {
      indices_buffer[i] = static_cast<float>((*indices)[i]);
    }
    auto& input_indices_packets =
        runner->MutableInputs()->Tag("INDICES").packets;
    input_indices_packets.push_back(mediapipe::Adopt(indices_tensors.release())
                                        .At(mediapipe::Timestamp(0)));
  }
}

// Compares the provided tensor contents with the expected values.
void ValidateResult(const Tensor& actual, const std::vector<float>& expected) {
  EXPECT_EQ(actual.element_type(), Tensor::ElementType::kFloat32);
  EXPECT_EQ(expected.size(), actual.shape().num_elements());
  auto view = actual.GetCpuReadView();
  auto buffer = view.buffer<float>();
  for (int i = 0; i < expected.size(); ++i) {
    EXPECT_FLOAT_EQ(expected[i], buffer[i]);
  }
}

TEST(ScoreCalibrationCalculatorTest, FailsWithNoSigmoid) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {}
    }
  )pb"));

  BuildGraph(&runner, {0.5, 0.5, 0.5});
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected at least one sigmoid, found none"));
}

TEST(ScoreCalibrationCalculatorTest, FailsWithNegativeScale) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { slope: 1 offset: 1 scale: -1 }
      }
    }
  )pb"));

  BuildGraph(&runner, {0.5, 0.5, 0.5});
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr("The scale parameter of the sigmoids must be positive"));
}

TEST(ScoreCalibrationCalculatorTest, FailsWithUnspecifiedTransformation) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { slope: 1 offset: 1 scale: 1 }
        score_transformation: UNSPECIFIED
      }
    }
  )pb"));

  BuildGraph(&runner, {0.5, 0.5, 0.5});
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Unsupported ScoreTransformation type"));
}

// Struct holding the parameters for parameterized tests below.
struct CalibrationTestParams {
  // The score transformation to apply.
  std::string score_transformation;
  // The expected results.
  std::vector<float> expected_results;
};

class CalibrationWithoutIndicesTest
    : public TestWithParam<CalibrationTestParams> {};

TEST_P(CalibrationWithoutIndicesTest, Succeeds) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(absl::StrFormat(
      R"pb(
        calculator: "ScoreCalibrationCalculator"
        input_stream: "SCORES:scores"
        output_stream: "CALIBRATED_SCORES:calibrated_scores"
        options {
          [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
            sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 }
            sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.3 }
            sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.6 }
            sigmoids {}
            score_transformation: %s
            default_score: 0.2
          }
        }
      )pb",
      GetParam().score_transformation)));

  BuildGraph(&runner, {0.2, 0.3, 0.4, 0.5});
  MP_ASSERT_OK(runner.Run());
  const Tensor& results = runner.Outputs()
                              .Get("CALIBRATED_SCORES", 0)
                              .packets[0]
                              .Get<std::vector<Tensor>>()[0];

  ValidateResult(results, GetParam().expected_results);
}

INSTANTIATE_TEST_SUITE_P(
    ScoreCalibrationCalculatorTest, CalibrationWithoutIndicesTest,
    Values(CalibrationTestParams{.score_transformation = "IDENTITY",
                                 .expected_results = {0.4948505976,
                                                      0.5059588508, 0.2, 0.2}},
           CalibrationTestParams{
               .score_transformation = "LOG",
               .expected_results = {0.2976901255, 0.3393665735, 0.2, 0.2}},
           CalibrationTestParams{
               .score_transformation = "INVERSE_LOGISTIC",
               .expected_results = {0.3203217641, 0.3778080605, 0.2, 0.2}}),
    [](const TestParamInfo<CalibrationWithoutIndicesTest::ParamType>& info) {
      return info.param.score_transformation;
    });

TEST(ScoreCalibrationCalculatorTest, FailsWithMissingSigmoids) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.3 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.6 }
        sigmoids {}
        score_transformation: LOG
        default_score: 0.2
      }
    }
  )pb"));

  BuildGraph(&runner, {0.2, 0.3, 0.4, 0.5, 0.6});
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Mismatch between number of sigmoids"));
}

TEST(ScoreCalibrationCalculatorTest, SucceedsWithIndices) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    input_stream: "INDICES:indices"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.3 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.6 }
        sigmoids {}
        score_transformation: IDENTITY
        default_score: 0.2
      }
    }
  )pb"));
  std::vector<int> indices = {1, 2, 3, 0};

  BuildGraph(&runner, {0.3, 0.4, 0.5, 0.2}, indices);
  MP_ASSERT_OK(runner.Run());
  const Tensor& results = runner.Outputs()
                              .Get("CALIBRATED_SCORES", 0)
                              .packets[0]
                              .Get<std::vector<Tensor>>()[0];
  ValidateResult(results, {0.5059588508, 0.2, 0.2, 0.4948505976});
}

TEST(ScoreCalibrationCalculatorTest, FailsWithNegativeIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    input_stream: "INDICES:indices"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.3 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.6 }
        sigmoids {}
        score_transformation: IDENTITY
        default_score: 0.2
      }
    }
  )pb"));
  std::vector<int> indices = {0, 1, 2, -1};

  BuildGraph(&runner, {0.2, 0.3, 0.4, 0.5}, indices);
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Expected positive indices"));
}

TEST(ScoreCalibrationCalculatorTest, FailsWithOutOfBoundsIndex) {
  CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "ScoreCalibrationCalculator"
    input_stream: "SCORES:scores"
    input_stream: "INDICES:indices"
    output_stream: "CALIBRATED_SCORES:calibrated_scores"
    options {
      [mediapipe.tasks.ScoreCalibrationCalculatorOptions.ext] {
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.3 }
        sigmoids { scale: 0.9 slope: 0.5 offset: 0.1 min_score: 0.6 }
        sigmoids {}
        score_transformation: IDENTITY
        default_score: 0.2
      }
    }
  )pb"));
  std::vector<int> indices = {0, 1, 5, 3};

  BuildGraph(&runner, {0.2, 0.3, 0.4, 0.5}, indices);
  auto status = runner.Run();

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr("Unable to get score calibration parameters for index"));
}

}  // namespace
}  // namespace mediapipe
