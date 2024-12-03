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

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"

namespace mediapipe {
namespace api2 {

using ::absl::StatusCode;
using ::mediapipe::tasks::CreateStatusWithPayload;
using ::mediapipe::tasks::MediaPipeTasksStatus;
using ::mediapipe::tasks::ScoreCalibrationCalculatorOptions;

namespace {
// Used to prevent log(<=0.0) in ClampedLog() calls.
constexpr float kLogScoreMinimum = 1e-16;

// Returns the following, depending on x:
//   x => threshold: log(x)
//   x < threshold: 2 * log(thresh) - log(2 * thresh - x)
// This form (a) is anti-symmetric about the threshold and (b) has continuous
// value and first derivative. This is done to prevent taking the log of values
// close to 0 which can lead to floating point errors and is better than simple
// clamping since it preserves order for scores less than the threshold.
float ClampedLog(float x, float threshold) {
  if (x < threshold) {
    return 2.0 * std::log(static_cast<double>(threshold)) -
           log(2.0 * threshold - x);
  }
  return std::log(static_cast<double>(x));
}
}  // namespace

// Applies score calibration to a tensor of score predictions, typically applied
// to the output of a classification or object detection model.
//
// See corresponding options for more details on the score calibration
// parameters and formula.
//
// Inputs:
//   SCORES - std::vector<Tensor>
//     A vector containing a single Tensor `x` of type kFloat32, representing
//     the scores to calibrate. By default (i.e. if INDICES is not connected),
//     x[i] will be calibrated using the sigmoid provided at index i in the
//     options.
//   INDICES - std::vector<Tensor> @Optional
//     An optional vector containing a single Tensor `y` of type kFloat32 and
//     same size as `x`. If provided, x[i] will be calibrated using the sigmoid
//     provided at index y[i] (casted as an integer) in the options. `x` and `y`
//     must contain the same number of elements. Typically used for object
//     detection models.
//
// Outputs:
//   CALIBRATED_SCORES - std::vector<Tensor>
//     A vector containing a single Tensor of type kFloat32 and of the same size
//     as the input tensors. Contains the output calibrated scores.
class ScoreCalibrationCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kScoresIn{"SCORES"};
  static constexpr Input<std::vector<Tensor>>::Optional kIndicesIn{"INDICES"};
  static constexpr Output<std::vector<Tensor>> kScoresOut{"CALIBRATED_SCORES"};
  MEDIAPIPE_NODE_CONTRACT(kScoresIn, kIndicesIn, kScoresOut);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  ScoreCalibrationCalculatorOptions options_;
  std::function<float(float)> score_transformation_;

  // Computes the calibrated score for the provided index. Does not check for
  // out-of-bounds index.
  float ComputeCalibratedScore(int index, float score);
  // Same as above, but does check for out-of-bounds index.
  absl::StatusOr<float> SafeComputeCalibratedScore(int index, float score);
};

absl::Status ScoreCalibrationCalculator::Open(CalculatorContext* cc) {
  options_ = cc->Options<ScoreCalibrationCalculatorOptions>();
  // Sanity checks.
  if (options_.sigmoids_size() == 0) {
    return CreateStatusWithPayload(StatusCode::kInvalidArgument,
                                   "Expected at least one sigmoid, found none.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  for (const auto& sigmoid : options_.sigmoids()) {
    if (sigmoid.has_scale() && sigmoid.scale() < 0.0) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("The scale parameter of the sigmoids must be "
                          "positive, found %f.",
                          sigmoid.scale()),
          MediaPipeTasksStatus::kInvalidArgumentError);
    }
  }
  // Set score transformation function once and for all.
  switch (options_.score_transformation()) {
    case tasks::ScoreCalibrationCalculatorOptions::IDENTITY:
      score_transformation_ = [](float x) { return x; };
      break;
    case tasks::ScoreCalibrationCalculatorOptions::LOG:
      score_transformation_ = [](float x) {
        return ClampedLog(x, kLogScoreMinimum);
      };
      break;
    case tasks::ScoreCalibrationCalculatorOptions::INVERSE_LOGISTIC:
      score_transformation_ = [](float x) {
        return (ClampedLog(x, kLogScoreMinimum) -
                ClampedLog(1.0 - x, kLogScoreMinimum));
      };
      break;
    default:
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Unsupported ScoreTransformation type: %s",
              ScoreCalibrationCalculatorOptions::ScoreTransformation_Name(
                  options_.score_transformation())),
          MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::Status ScoreCalibrationCalculator::Process(CalculatorContext* cc) {
  RET_CHECK_EQ(kScoresIn(cc)->size(), 1);
  const auto& scores = (*kScoresIn(cc))[0];
  RET_CHECK(scores.element_type() == Tensor::ElementType::kFloat32);
  auto scores_view = scores.GetCpuReadView();
  const float* raw_scores = scores_view.buffer<float>();
  int num_scores = scores.shape().num_elements();

  auto output_tensors = std::make_unique<std::vector<Tensor>>();
  output_tensors->reserve(1);
  output_tensors->emplace_back(scores.element_type(), scores.shape());
  auto calibrated_scores = &output_tensors->back();
  auto calibrated_scores_view = calibrated_scores->GetCpuWriteView();
  float* raw_calibrated_scores = calibrated_scores_view.buffer<float>();

  if (kIndicesIn(cc).IsConnected()) {
    RET_CHECK_EQ(kIndicesIn(cc)->size(), 1);
    const auto& indices = (*kIndicesIn(cc))[0];
    RET_CHECK(indices.element_type() == Tensor::ElementType::kFloat32);
    if (num_scores != indices.shape().num_elements()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Mismatch between number of elements in the input "
                          "scores tensor (%d) and indices tensor (%d).",
                          num_scores, indices.shape().num_elements()),
          MediaPipeTasksStatus::kMetadataInconsistencyError);
    }
    auto indices_view = indices.GetCpuReadView();
    const float* raw_indices = indices_view.buffer<float>();
    for (int i = 0; i < num_scores; ++i) {
      // Use the "safe" flavor as we need to check that the externally provided
      // indices are not out-of-bounds.
      MP_ASSIGN_OR_RETURN(raw_calibrated_scores[i],
                          SafeComputeCalibratedScore(
                              static_cast<int>(raw_indices[i]), raw_scores[i]));
    }
  } else {
    if (num_scores != options_.sigmoids_size()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Mismatch between number of sigmoids (%d) and number "
                          "of elements in the input scores tensor (%d).",
                          options_.sigmoids_size(), num_scores),
          MediaPipeTasksStatus::kMetadataInconsistencyError);
    }
    for (int i = 0; i < num_scores; ++i) {
      // Use the "unsafe" flavor as we have already checked for out-of-bounds
      // issues.
      raw_calibrated_scores[i] = ComputeCalibratedScore(i, raw_scores[i]);
    }
  }
  kScoresOut(cc).Send(std::move(output_tensors));
  return absl::OkStatus();
}

float ScoreCalibrationCalculator::ComputeCalibratedScore(int index,
                                                         float score) {
  const auto& sigmoid = options_.sigmoids(index);

  bool is_empty =
      !sigmoid.has_scale() || !sigmoid.has_offset() || !sigmoid.has_slope();
  bool is_below_min_score =
      sigmoid.has_min_score() && score < sigmoid.min_score();
  if (is_empty || is_below_min_score) {
    return options_.default_score();
  }

  float transformed_score = score_transformation_(score);
  float scale_shifted_score =
      transformed_score * sigmoid.slope() + sigmoid.offset();
  // For numerical stability use 1 / (1+exp(-x)) when scale_shifted_score >= 0
  // and exp(x) / (1+exp(x)) when scale_shifted_score < 0.
  float calibrated_score;
  if (scale_shifted_score >= 0.0) {
    calibrated_score =
        sigmoid.scale() /
        (1.0 + std::exp(static_cast<double>(-scale_shifted_score)));
  } else {
    float score_exp = std::exp(static_cast<double>(scale_shifted_score));
    calibrated_score = sigmoid.scale() * score_exp / (1.0 + score_exp);
  }
  // Scale is non-negative (checked in SigmoidFromLabelAndLine),
  // thus calibrated_score should be in the range of [0, scale]. However, due to
  // numberical stability issue, it may fall out of the boundary. Cap the value
  // to [0, scale] instead.
  return std::max(std::min(calibrated_score, sigmoid.scale()), 0.0f);
}

absl::StatusOr<float> ScoreCalibrationCalculator::SafeComputeCalibratedScore(
    int index, float score) {
  if (index < 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected positive indices, found %d.", index),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (index > options_.sigmoids_size()) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Unable to get score calibration parameters for index "
                        "%d : only %d sigmoids were provided.",
                        index, options_.sigmoids_size()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }
  return ComputeCalibratedScore(index, score);
}

MEDIAPIPE_REGISTER_NODE(ScoreCalibrationCalculator);

}  // namespace api2
}  // namespace mediapipe
