/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

constexpr char kHandLandmarksTag[] = "HAND_LANDMARKS";
constexpr char kHandWorldLandmarksTag[] = "HAND_WORLD_LANDMARKS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
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
absl::StatusOr<LandmarkListT> CanonicalizeOffsetAndScale(
    const LandmarkListT& landmarks) {
  if (landmarks.landmark_size() == 0) {
    return ::absl::InvalidArgumentError(
        "Expected non-zero number of input landmarks.");
  }
  LandmarkListT canonicalized_landmarks;
  const auto& wrist = landmarks.landmark(0);
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::min();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::min();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& old_landmark = landmarks.landmark(i);
    auto* new_landmark = canonicalized_landmarks.add_landmark();
    new_landmark->set_x(old_landmark.x() - wrist.x());
    new_landmark->set_y(old_landmark.y() - wrist.y());
    new_landmark->set_z(old_landmark.z() - wrist.z());
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

template <class LandmarkListT>
absl::Status ProcessLandmarks(LandmarkListT hand_landmarks, bool is_normalized,
                              CalculatorContext* cc) {
  const bool normalize_wrt_aspect_ratio =
      is_normalized && !cc->Inputs().Tag(kImageSizeTag).IsEmpty();

  if (normalize_wrt_aspect_ratio) {
    const auto [width, height] =
        cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
    ASSIGN_OR_RETURN(hand_landmarks, NormalizeLandmarkAspectRatio(
                                         hand_landmarks, width, height));
  }

  ASSIGN_OR_RETURN(auto canonicalized_landmarks,
                   CanonicalizeOffsetAndScale(hand_landmarks));
  auto landmarks_matrix = std::make_unique<Matrix>();
  *landmarks_matrix = LandmarksToMatrix(canonicalized_landmarks);
  cc->Outputs()
      .Tag(kLandmarksMatrixTag)
      .Add(landmarks_matrix.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace

// Convert single hand landmarks into a matrix. The landmarks are normalized
// w.r.t. the image's aspect ratio and w.r.t the wrist. This pre-processing step
// is required for the hand gesture recognition model.
//
// Input:
//   HAND_LANDMARKS - Single hand landmarks. Use *either* HAND_LANDMARKS or
//                    HAND_WORLD_LANDMARKS.
//   HAND_WORLD_LANDMARKS - Single hand world 3d landmarks. Use *either*
//                          HAND_LANDMARKS or HAND_WORLD_LANDMARKS.
//   IMAGE_SIZE - (width, height) of the image
// Output:
//   LANDMARKS_MATRIX - Matrix for hand landmarks.
//
// Usage example:
// node {
//   calculator: "HandLandmarksToMatrixCalculator"
//   input_stream: "HAND_LANDMARKS:hand_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "LANDMARKS_MATRIX:landmarks_matrix"
// }
class HandLandmarksToMatrixCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs()
        .Tag(kHandLandmarksTag)
        .Set<NormalizedLandmarkList>()
        .Optional();
    cc->Inputs().Tag(kHandWorldLandmarksTag).Set<LandmarkList>().Optional();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>().Optional();
    cc->Outputs().Tag(kLandmarksMatrixTag).Set<Matrix>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    RET_CHECK(cc->Inputs().HasTag(kHandLandmarksTag) ^
              cc->Inputs().HasTag(kHandWorldLandmarksTag));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override;
};

REGISTER_CALCULATOR(HandLandmarksToMatrixCalculator);

absl::Status HandLandmarksToMatrixCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kHandLandmarksTag)) {
    if (!cc->Inputs().Tag(kHandLandmarksTag).IsEmpty()) {
      auto hand_landmarks =
          cc->Inputs().Tag(kHandLandmarksTag).Get<NormalizedLandmarkList>();
      return ProcessLandmarks(hand_landmarks, /*is_normalized=*/true, cc);
    }
  } else if (cc->Inputs().HasTag(kHandWorldLandmarksTag)) {
    if (!cc->Inputs().Tag(kHandWorldLandmarksTag).IsEmpty()) {
      auto hand_landmarks =
          cc->Inputs().Tag(kHandWorldLandmarksTag).Get<LandmarkList>();
      return ProcessLandmarks(hand_landmarks, /*is_normalized=*/false, cc);
    }
  }
  return absl::OkStatus();
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
