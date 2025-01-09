/* Copyright 2022 The MediaPipe Authors.

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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/landmarks_to_matrix_calculator.pb.h"

// TODO Update to use API2
namespace mediapipe {
namespace api2 {

using ::mediapipe::NormalizedRect;

namespace {

constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kLandmarksMatrixTag[] = "LANDMARKS_MATRIX";
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
absl::Status ProcessLandmarks(LandmarkListT landmarks, CalculatorContext* cc) {
  if (IsNormalized<LandmarkListT>()) {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag) &&
              !cc->Inputs().Tag(kImageSizeTag).IsEmpty());
    const auto [width, height] =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    MP_ASSIGN_OR_RETURN(landmarks,
                        NormalizeLandmarkAspectRatio(landmarks, width, height));
  }

  if (cc->Inputs().HasTag(kNormRectTag)) {
    RET_CHECK(!cc->Inputs().Tag(kNormRectTag).IsEmpty());
    const auto rotation =
        cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>().rotation();
    MP_ASSIGN_OR_RETURN(landmarks, RotateLandmarks(landmarks, rotation));
  }

  const auto& options = cc->Options<LandmarksToMatrixCalculatorOptions>();
  if (options.object_normalization()) {
    MP_ASSIGN_OR_RETURN(
        landmarks,
        NormalizeObject(landmarks,
                        options.object_normalization_origin_offset()));
  }

  auto landmarks_matrix = std::make_unique<Matrix>();
  *landmarks_matrix = LandmarksToMatrix(landmarks);
  cc->Outputs()
      .Tag(kLandmarksMatrixTag)
      .Add(landmarks_matrix.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace

// Convert landmarks into a matrix. The landmarks are normalized
// w.r.t. the image's aspect ratio (if they are NormalizedLandmarksList)
// and optionally w.r.t and "origin" landmark. This pre-processing step
// is required for the some models.
//
// Input:
//   LANDMARKS - Landmarks of one object. Use *either* LANDMARKS or
//               WORLD_LANDMARKS.
//   WORLD_LANDMARKS - World 3d landmarks of one object. Use *either*
//               LANDMARKS or WORLD_LANDMARKS.
//   IMAGE_SIZE - (width, height) of the image
//   NORM_RECT - Optional NormalizedRect object whose 'rotation' field is used
//               to rotate the landmarks.
// Output:
//   LANDMARKS_MATRIX - Matrix for the landmarks.
//
// Usage example:
// node {
//   calculator: "LandmarksToMatrixCalculator"
//   input_stream: "LANDMARKS:hand_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
//   options {
//     [mediapipe.LandmarksToMatrixCalculatorOptions.ext] {
//       object_normalization: true
//       object_normalization_origin_offset: 0
//     }
//   }
// }
class LandmarksToMatrixCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>().Optional();
    cc->Inputs().Tag(kWorldLandmarksTag).Set<LandmarkList>().Optional();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>().Optional();
    cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>().Optional();
    cc->Outputs().Tag(kLandmarksMatrixTag).Set<Matrix>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ^
              cc->Inputs().HasTag(kWorldLandmarksTag));
    const auto& options = cc->Options<LandmarksToMatrixCalculatorOptions>();
    RET_CHECK(options.has_object_normalization());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(LandmarksToMatrixCalculator);

absl::Status LandmarksToMatrixCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kLandmarksTag)) {
    if (!cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
      auto landmarks =
          cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();
      return ProcessLandmarks(landmarks, cc);
    }
  } else if (cc->Inputs().HasTag(kWorldLandmarksTag)) {
    if (!cc->Inputs().Tag(kWorldLandmarksTag).IsEmpty()) {
      auto landmarks = cc->Inputs().Tag(kWorldLandmarksTag).Get<LandmarkList>();
      return ProcessLandmarks(landmarks, cc);
    }
  }
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
