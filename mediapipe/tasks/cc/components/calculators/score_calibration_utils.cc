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

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {

namespace {
// Converts ScoreTransformation type from TFLite Metadata to calculator options.
ScoreCalibrationCalculatorOptions::ScoreTransformation
ConvertScoreTransformationType(tflite::ScoreTransformationType type) {
  switch (type) {
    case tflite::ScoreTransformationType_IDENTITY:
      return ScoreCalibrationCalculatorOptions::IDENTITY;
    case tflite::ScoreTransformationType_LOG:
      return ScoreCalibrationCalculatorOptions::LOG;
    case tflite::ScoreTransformationType_INVERSE_LOGISTIC:
      return ScoreCalibrationCalculatorOptions::INVERSE_LOGISTIC;
  }
}

// Parses a single line of the score calibration file into the provided sigmoid.
absl::Status FillSigmoidFromLine(
    absl::string_view line,
    ScoreCalibrationCalculatorOptions::Sigmoid* sigmoid) {
  if (line.empty()) {
    return absl::OkStatus();
  }
  std::vector<absl::string_view> str_params = absl::StrSplit(line, ',');
  if (str_params.size() != 3 && str_params.size() != 4) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Expected 3 or 4 parameters per line in score "
                        "calibration file, got %d.",
                        str_params.size()),
        MediaPipeTasksStatus::kMetadataMalformedScoreCalibrationError);
  }
  std::vector<float> params(str_params.size());
  for (int i = 0; i < str_params.size(); ++i) {
    if (!absl::SimpleAtof(str_params[i], &params[i])) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Could not parse score calibration parameter as float: %s.",
              str_params[i]),
          MediaPipeTasksStatus::kMetadataMalformedScoreCalibrationError);
    }
  }
  if (params[0] < 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "The scale parameter of the sigmoids must be positive, found %f.",
            params[0]),
        MediaPipeTasksStatus::kMetadataMalformedScoreCalibrationError);
  }
  sigmoid->set_scale(params[0]);
  sigmoid->set_slope(params[1]);
  sigmoid->set_offset(params[2]);
  if (params.size() == 4) {
    sigmoid->set_min_score(params[3]);
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status ConfigureScoreCalibration(
    tflite::ScoreTransformationType score_transformation, float default_score,
    absl::string_view score_calibration_file,
    ScoreCalibrationCalculatorOptions* calculator_options) {
  calculator_options->set_score_transformation(
      ConvertScoreTransformationType(score_transformation));
  calculator_options->set_default_score(default_score);

  if (score_calibration_file.empty()) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Expected non-empty score calibration file.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  std::vector<absl::string_view> lines =
      absl::StrSplit(score_calibration_file, '\n');
  for (const auto& line : lines) {
    auto* sigmoid = calculator_options->add_sigmoids();
    MP_RETURN_IF_ERROR(FillSigmoidFromLine(line, sigmoid));
  }

  return absl::OkStatus();
}

}  // namespace tasks
}  // namespace mediapipe
