// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/landmarks_transformation_calculator.h"

#include <utility>

#include "mediapipe/calculators/util/landmarks_transformation_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {
namespace api2 {

namespace {

StatusOr<LandmarkList> NormalizeTranslation(const LandmarkList& in_landmarks) {
  RET_CHECK_GT(in_landmarks.landmark_size(), 0);

  double x_sum = 0.0f;
  double y_sum = 0.0f;
  double z_sum = 0.0f;
  for (auto& in_landmark : in_landmarks.landmark()) {
    x_sum += in_landmark.x();
    y_sum += in_landmark.y();
    z_sum += in_landmark.z();
  }

  float x_mean = x_sum / in_landmarks.landmark_size();
  float y_mean = y_sum / in_landmarks.landmark_size();
  float z_mean = z_sum / in_landmarks.landmark_size();

  LandmarkList out_landmarks;
  for (auto& in_landmark : in_landmarks.landmark()) {
    auto* out_landmark = out_landmarks.add_landmark();
    *out_landmark = in_landmark;
    out_landmark->set_x(in_landmark.x() - x_mean);
    out_landmark->set_y(in_landmark.y() - y_mean);
    out_landmark->set_z(in_landmark.z() - z_mean);
  }

  return out_landmarks;
}

StatusOr<LandmarkList> FlipAxis(
    const LandmarkList& in_landmarks,
    const LandmarksTransformationCalculatorOptions::FlipAxis& options) {
  float x_mul = options.flip_x() ? -1 : 1;
  float y_mul = options.flip_y() ? -1 : 1;
  float z_mul = options.flip_z() ? -1 : 1;

  LandmarkList out_landmarks;
  for (auto& in_landmark : in_landmarks.landmark()) {
    auto* out_landmark = out_landmarks.add_landmark();
    *out_landmark = in_landmark;
    out_landmark->set_x(in_landmark.x() * x_mul);
    out_landmark->set_y(in_landmark.y() * y_mul);
    out_landmark->set_z(in_landmark.z() * z_mul);
  }

  return out_landmarks;
}

}  // namespace

class LandmarksTransformationCalculatorImpl
    : public NodeImpl<LandmarksTransformationCalculator> {
 public:
  static absl::Status UpdateContract(CalculatorContract* cc) {
    // Check that if options input stream is connected there should be no static
    // options in calculator. Currently there is no such functionality, so we'll
    // just check for the number of transforms.
    if (kInOptions(cc).IsConnected()) {
      RET_CHECK_EQ(cc->Options<LandmarksTransformationCalculatorOptions>()
                       .transformation_size(),
                   0);
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<LandmarksTransformationCalculatorOptions>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (kInLandmarks(cc).IsEmpty()) {
      return absl::OkStatus();
    }

    // Get transformation options for either calculator parameters or input
    // stream. Input stream has higher priority.
    LandmarksTransformationCalculatorOptions options;
    if (kInOptions(cc).IsConnected()) {
      // If input stream is connected but is empty - use no transformations and
      // return landmarks as is.
      if (!kInOptions(cc).IsEmpty()) {
        options = kInOptions(cc).Get();
      }
    } else {
      options = options_;
    }

    LandmarkList landmarks = kInLandmarks(cc).Get();

    for (auto& transformation : options.transformation()) {
      if (transformation.has_normalize_translation()) {
        MP_ASSIGN_OR_RETURN(landmarks, NormalizeTranslation(landmarks));
      } else if (transformation.has_flip_axis()) {
        MP_ASSIGN_OR_RETURN(landmarks,
                            FlipAxis(landmarks, transformation.flip_axis()));
      } else {
        RET_CHECK_FAIL() << "Unknown landmarks transformation";
      }
    }

    kOutLandmarks(cc).Send(std::move(landmarks));

    return absl::OkStatus();
  }

 private:
  LandmarksTransformationCalculatorOptions options_;
};
MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksTransformationCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
