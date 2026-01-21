/* Copyright 2025 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/api3/calculator_contract.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.pb.h"

namespace mediapipe {
namespace tasks {

using ::mediapipe::NormalizedRect;
using ::mediapipe::api3::Calculator;
using ::mediapipe::api3::CalculatorContext;
using ::mediapipe::api3::CalculatorContract;
using ::mediapipe::tasks::LandmarksToMatrixNode;

namespace {

constexpr int kFeaturesPerLandmark = 3;

template <class LandmarkListT>
absl::StatusOr<LandmarkListT> NormalizeLandmarkAspectRatio(
    const LandmarkListT& landmarks, float width, float height) {
  const float max_dim = std::max(width, height);
  if (max_dim <= 0) {
    return ::absl::InvalidArgumentError(
        absl::StrCat("Invalid image dimensions: [", width, ",", height, "]"));
  }
  const float width_scale_factor = width / max_dim;
  const float height_scale_factor = height / max_dim;
  LandmarkListT normalized_landmarks;
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& old_landmark = landmarks.landmark(i);
    auto* new_landmark = normalized_landmarks.add_landmark();
    new_landmark->set_x((old_landmark.x() - 0.5) * width_scale_factor + 0.5);
    new_landmark->set_y((old_landmark.y() - 0.5) * height_scale_factor + 0.5);
    new_landmark->set_z(old_landmark.z());
  }
  return normalized_landmarks;
}

template <class LandmarkListT>
absl::StatusOr<LandmarkListT> RotateLandmarks(const LandmarkListT& landmarks,
                                              float rotation) {
  float cos = std::cos(rotation);
  // Negate because Y-axis points down and not up.
  float sin = std::sin(-rotation);
  LandmarkListT rotated_landmarks;
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& old_landmark = landmarks.landmark(i);
    float x = old_landmark.x() - 0.5;
    float y = old_landmark.y() - 0.5;
    auto* new_landmark = rotated_landmarks.add_landmark();
    new_landmark->set_x(x * cos - y * sin + 0.5);
    new_landmark->set_y(y * cos + x * sin + 0.5);
    new_landmark->set_z(old_landmark.z());
  }
  return rotated_landmarks;
}

template <class LandmarkListT>
absl::StatusOr<LandmarkListT> NormalizeObject(const LandmarkListT& landmarks,
                                              int origin_offset) {
  if (landmarks.landmark_size() == 0) {
    return ::absl::InvalidArgumentError(
        "Expected non-zero number of input landmarks.");
  }
  LandmarkListT canonicalized_landmarks;
  const auto& origin = landmarks.landmark(origin_offset);
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::min();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& old_landmark = landmarks.landmark(i);
    auto* new_landmark = canonicalized_landmarks.add_landmark();
    new_landmark->set_x(old_landmark.x() - origin.x());
    new_landmark->set_y(old_landmark.y() - origin.y());
    new_landmark->set_z(old_landmark.z() - origin.z());
    min_x = std::min(min_x, new_landmark->x());
    max_x = std::max(max_x, new_landmark->x());
    min_y = std::min(min_y, new_landmark->y());
    max_y = std::max(max_y, new_landmark->y());
  }
  const float kEpsilon = 1e-5;
  const float scale = std::max(max_x - min_x, max_y - min_y) + kEpsilon;
  for (auto& landmark : *canonicalized_landmarks.mutable_landmark()) {
    landmark.set_x(landmark.x() / scale);
    landmark.set_y(landmark.y() / scale);
    landmark.set_z(landmark.z() / scale);
  }
  return canonicalized_landmarks;
}

template <class LandmarkListT>
Matrix LandmarksToMatrix(const LandmarkListT& landmarks) {
  auto matrix = Matrix(kFeaturesPerLandmark, landmarks.landmark_size());
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    matrix(0, i) = landmark.x();
    matrix(1, i) = landmark.y();
    matrix(2, i) = landmark.z();
  }
  return matrix;
}

template <class T>
struct DependentFalse : std::false_type {};

template <typename LandmarkListT>
bool IsNormalized() {
  if constexpr (std::is_same_v<LandmarkListT, NormalizedLandmarkList>) {
    return true;
  } else if constexpr (std::is_same_v<LandmarkListT, LandmarkList>) {
    return false;
  } else {
    static_assert(DependentFalse<LandmarkListT>::value,
                  "Type is not supported.");
  }
}

template <class LandmarkListT>
absl::Status ProcessLandmarks(LandmarkListT landmarks,
                              CalculatorContext<LandmarksToMatrixNode>& cc) {
  if (IsNormalized<LandmarkListT>()) {
    RET_CHECK(cc.image_size.IsConnected() && cc.image_size);
    const auto& [width, height] = cc.image_size.GetOrDie();
    MP_ASSIGN_OR_RETURN(landmarks,
                        NormalizeLandmarkAspectRatio(landmarks, width, height));
  }

  if (cc.norm_rect.IsConnected()) {
    RET_CHECK(cc.norm_rect);
    const NormalizedRect& norm_rect = cc.norm_rect.GetOrDie();
    const float rotation = norm_rect.rotation();
    MP_ASSIGN_OR_RETURN(landmarks, RotateLandmarks(landmarks, rotation));
  }

  const LandmarksToMatrixCalculatorOptions& options = cc.options.Get();
  if (options.object_normalization()) {
    MP_ASSIGN_OR_RETURN(
        landmarks,
        NormalizeObject(landmarks,
                        options.object_normalization_origin_offset()));
  }

  auto landmarks_matrix = std::make_unique<Matrix>();
  *landmarks_matrix = LandmarksToMatrix(landmarks);
  cc.landmarks_matrix.Send(std::move(landmarks_matrix));
  return absl::OkStatus();
}

}  // namespace

class LandmarksToMatrixCalculatorImpl
    : public Calculator<LandmarksToMatrixNode,
                        LandmarksToMatrixCalculatorImpl> {
 public:
  static absl::Status UpdateContract(
      CalculatorContract<LandmarksToMatrixNode>& cc) {
    RET_CHECK(cc.landmarks.IsConnected() ^ cc.world_landmarks.IsConnected());
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext<LandmarksToMatrixNode>& cc) override {
    const LandmarksToMatrixCalculatorOptions& options = cc.options.Get();
    RET_CHECK(options.has_object_normalization());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<LandmarksToMatrixNode>& cc) override;
};

absl::Status LandmarksToMatrixCalculatorImpl::Process(
    CalculatorContext<LandmarksToMatrixNode>& cc) {
  if (cc.landmarks.IsConnected()) {
    if (!cc.landmarks) return absl::OkStatus();
    const NormalizedLandmarkList& landmarks = cc.landmarks.GetOrDie();
    return ProcessLandmarks(landmarks, cc);
  } else if (cc.world_landmarks.IsConnected()) {
    if (!cc.world_landmarks) return absl::OkStatus();
    const LandmarkList& world_landmarks = cc.world_landmarks.GetOrDie();
    return ProcessLandmarks(world_landmarks, cc);
  }
  return absl::OkStatus();
}

}  // namespace tasks
}  // namespace mediapipe
