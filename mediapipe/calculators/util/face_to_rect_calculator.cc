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

#include <algorithm>
#include <cmath>

#include "absl/status/status.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.h"
#include "mediapipe/calculators/util/face_to_rect_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/vector.h"

namespace mediapipe {

// A calculator to convert face detection proto to mediapipe rect.
// node {
//   calculator: "FaceToRectCalculator"
//   input_stream: "DETECTION:detection"
//   input_stream: "IMAGE_SIZE:frame_size"
//   output_stream: "NORM_RECT:rect"
//   node_options: {
//     [type.googleapis.com/mediapipe.FaceToRectCalculatorOptions] {
//       eye_landmark_size: 1
//       nose_landmark_size: 2
//       mouth_landmark_size: 2
//       eye_to_mouth_scale: 3.42
//       eye_to_eye_scale: 3.8
//     }
//   }
// }
//
class FaceToRectCalculator : public DetectionsToRectsCalculator {
 public:
  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));

    // Default values.
    face_options_.set_eye_landmark_size(2);
    face_options_.set_nose_landmark_size(0);
    face_options_.set_mouth_landmark_size(2);
    face_options_.set_eye_to_mouth_mix(0.1f);
    face_options_.set_eye_to_mouth_scale(3.6f);
    face_options_.set_eye_to_eye_scale(4.0f);
    face_options_.MergeFrom(cc->Options<FaceToRectCalculatorOptions>());

    RET_CHECK(face_options_.eye_landmark_size() > 0 &&
              face_options_.mouth_landmark_size() > 0)
        << "Eye landmarks and mouth landmarks cannot be empty.";

    total_landmarks_ = face_options_.eye_landmark_size() * 2 +
                       face_options_.nose_landmark_size() +
                       face_options_.mouth_landmark_size();

    rotate_ = true;
    return absl::OkStatus();
  }

 private:
  absl::Status DetectionToRect(const Detection& detection,
                               const DetectionSpec& detection_spec,
                               Rect* rect) override {
    const int width = detection_spec.image_size->first;
    const int height = detection_spec.image_size->second;
    return ComputeFaceRect(detection, width, height, rect);
  }

  absl::Status DetectionToNormalizedRect(const Detection& detection,
                                         const DetectionSpec& detection_spec,
                                         NormalizedRect* rect) override {
    const int width = detection_spec.image_size->first;
    const int height = detection_spec.image_size->second;
    Rect rect_pix;
    MP_RETURN_IF_ERROR(ComputeFaceRect(detection, width, height, &rect_pix));

    const float width_recip = 1.f / width;
    const float height_recip = 1.f / height;
    rect->set_x_center(rect_pix.x_center() * width_recip);
    rect->set_y_center(rect_pix.y_center() * height_recip);
    rect->set_width(rect_pix.width() * width_recip);
    rect->set_height(rect_pix.height() * height_recip);
    return absl::OkStatus();
  }

  absl::Status ComputeRotation(const Detection& detection,
                               const DetectionSpec& detection_spec,
                               float* rotation) override {
    // eye_to_eye_ and eye_to_mouth_ are computed in ComputeFaceRect
    Vector2_f dir =
        eye_to_eye_ - Vector2_f(-eye_to_mouth_.y(), eye_to_mouth_.x());
    *rotation = NormalizeRadians(target_angle_ + std::atan2(dir.y(), dir.x()));
    return absl::OkStatus();
  }

  // Compute a face rectangle from detection landmarks.
  absl::Status ComputeFaceRect(const Detection& detection, int width,
                               int height, Rect* rect) {
    Vector2_f left_eye(0.f, 0.f), right_eye(0.f, 0.f), mouth(0.f, 0.f);
    if (!GetLandmarks(detection, width, height, &left_eye, &right_eye,
                      &mouth)) {
      return absl::InvalidArgumentError(
          "Detection has wrong number of keypoints.");
    }

    const Vector2_f eye_center = (left_eye + right_eye) * 0.5f;
    eye_to_eye_ = right_eye - left_eye;
    eye_to_mouth_ = mouth - eye_center;
    const Vector2_f center =
        eye_center + eye_to_mouth_ * face_options_.eye_to_mouth_mix();

    rect->set_x_center(std::round(center.x()));
    rect->set_y_center(std::round(center.y()));

    const float scale =
        std::max(eye_to_mouth_.Norm() * face_options_.eye_to_mouth_scale(),
                 eye_to_eye_.Norm() * face_options_.eye_to_eye_scale());
    rect->set_width(std::round(scale));
    rect->set_height(std::round(scale));
    return absl::OkStatus();
  }

  // Gets eyes and mouth landmarks from a face detection.
  bool GetLandmarks(const Detection& detection, int width, int height,
                    Vector2_f* left_eye, Vector2_f* right_eye,
                    Vector2_f* mouth);

  FaceToRectCalculatorOptions face_options_;
  int total_landmarks_ = 0;
  Vector2_f eye_to_eye_;
  Vector2_f eye_to_mouth_;
};

REGISTER_CALCULATOR(FaceToRectCalculator);

bool FaceToRectCalculator::GetLandmarks(const Detection& detection, int width,
                                        int height, Vector2_f* left_eye,
                                        Vector2_f* right_eye,
                                        Vector2_f* mouth) {
  const auto& location_data = detection.location_data();
  if (location_data.relative_keypoints_size() < total_landmarks_) {
    return false;
  }

  // eyes
  Vector2_f le(0.f, 0.f);
  Vector2_f re(0.f, 0.f);
  int i = 0;
  for (; i < face_options_.eye_landmark_size(); ++i) {
    const auto& left_point = location_data.relative_keypoints(i);
    le += Vector2_f(left_point.x() * width, left_point.y() * height);
    const auto& right_point =
        location_data.relative_keypoints(i + face_options_.eye_landmark_size());
    re += Vector2_f(right_point.x() * width, right_point.y() * height);
  }
  *left_eye = le / face_options_.eye_landmark_size();
  *right_eye = re / face_options_.eye_landmark_size();

  // mouth
  Vector2_f m(0.f, 0.f);
  i += face_options_.eye_landmark_size() + face_options_.nose_landmark_size();
  for (int j = 0; j < face_options_.mouth_landmark_size(); ++j) {
    const auto& point = location_data.relative_keypoints(i + j);
    m += Vector2_f(point.x() * width, point.y() * height);
  }
  *mouth = m / face_options_.mouth_landmark_size();
  return true;
}

}  // namespace mediapipe
