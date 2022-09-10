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

#include "mediapipe/tasks/cc/components/calculators/score_calibration_utils.h"

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace {

using ::testing::HasSubstr;

TEST(ConfigureScoreCalibrationTest, SucceedsWithoutTrailingNewline) {
  ScoreCalibrationCalculatorOptions options;
  std::string score_calibration_file =
      absl::StrCat("\n", "0.1,0.2,0.3\n", "0.4,0.5,0.6,0.7");

  MP_ASSERT_OK(ConfigureScoreCalibration(
      tflite::ScoreTransformationType_IDENTITY,
      /*default_score=*/0.5, score_calibration_file, &options));

  EXPECT_THAT(
      options,
      EqualsProto(ParseTextProtoOrDie<ScoreCalibrationCalculatorOptions>(R"pb(
        score_transformation: IDENTITY
        default_score: 0.5
        sigmoids {}
        sigmoids { scale: 0.1 slope: 0.2 offset: 0.3 }
        sigmoids { scale: 0.4 slope: 0.5 offset: 0.6 min_score: 0.7 }
      )pb")));
}

TEST(ConfigureScoreCalibrationTest, SucceedsWithTrailingNewline) {
  ScoreCalibrationCalculatorOptions options;
  std::string score_calibration_file =
      absl::StrCat("\n", "0.1,0.2,0.3\n", "0.4,0.5,0.6,0.7\n");

  MP_ASSERT_OK(ConfigureScoreCalibration(tflite::ScoreTransformationType_LOG,
                                         /*default_score=*/0.5,
                                         score_calibration_file, &options));

  EXPECT_THAT(
      options,
      EqualsProto(ParseTextProtoOrDie<ScoreCalibrationCalculatorOptions>(R"pb(
        score_transformation: LOG
        default_score: 0.5
        sigmoids {}
        sigmoids { scale: 0.1 slope: 0.2 offset: 0.3 }
        sigmoids { scale: 0.4 slope: 0.5 offset: 0.6 min_score: 0.7 }
        sigmoids {}
      )pb")));
}

TEST(ConfigureScoreCalibrationTest, FailsWithEmptyFile) {
  ScoreCalibrationCalculatorOptions options;

  auto status =
      ConfigureScoreCalibration(tflite::ScoreTransformationType_LOG,
                                /*default_score=*/0.5,
                                /*score_calibration_file=*/"", &options);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected non-empty score calibration file"));
}

TEST(ConfigureScoreCalibrationTest, FailsWithInvalidNumParameters) {
  ScoreCalibrationCalculatorOptions options;
  std::string score_calibration_file = absl::StrCat("0.1,0.2,0.3\n", "0.1,0.2");

  auto status = ConfigureScoreCalibration(tflite::ScoreTransformationType_LOG,
                                          /*default_score=*/0.5,
                                          score_calibration_file, &options);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("Expected 3 or 4 parameters per line"));
}

TEST(ConfigureScoreCalibrationTest, FailsWithNonParseableParameter) {
  ScoreCalibrationCalculatorOptions options;
  std::string score_calibration_file =
      absl::StrCat("0.1,0.2,0.3\n", "0.1,foo,0.3\n");

  auto status = ConfigureScoreCalibration(tflite::ScoreTransformationType_LOG,
                                          /*default_score=*/0.5,
                                          score_calibration_file, &options);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr("Could not parse score calibration parameter as float"));
}

TEST(ConfigureScoreCalibrationTest, FailsWithNegativeScaleParameter) {
  ScoreCalibrationCalculatorOptions options;
  std::string score_calibration_file =
      absl::StrCat("0.1,0.2,0.3\n", "-0.1,0.2,0.3\n");

  auto status = ConfigureScoreCalibration(tflite::ScoreTransformationType_LOG,
                                          /*default_score=*/0.5,
                                          score_calibration_file, &options);

  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.message(),
      HasSubstr("The scale parameter of the sigmoids must be positive"));
}

}  // namespace
}  // namespace tasks
}  // namespace mediapipe
