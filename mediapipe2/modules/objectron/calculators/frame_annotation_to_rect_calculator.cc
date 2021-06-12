// Copyright 2020 The MediaPipe Authors.
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

#include <cmath>
#include <vector>

#include "Eigen/Dense"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/frame_annotation_to_rect_calculator.pb.h"

namespace mediapipe {

using Matrix3fRM = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using Eigen::Vector2f;
using Eigen::Vector3f;

namespace {

constexpr char kInputFrameAnnotationTag[] = "FRAME_ANNOTATION";
constexpr char kOutputNormRectsTag[] = "NORM_RECTS";

}  // namespace

// A calculator that converts FrameAnnotation proto to NormalizedRect.
// The rotation angle of the NormalizedRect is derived from object's 3d pose.
// The angle is calculated such that after rotation the 2d projection of y-axis.
// on the image plane is always vertical.
class FrameAnnotationToRectCalculator : public CalculatorBase {
 public:
  enum ViewStatus {
    TOP_VIEW_ON,
    TOP_VIEW_OFF,
  };

  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  void AddAnnotationToRect(const ObjectAnnotation& annotation,
                           std::vector<NormalizedRect>* rect);
  float RotationAngleFromAnnotation(const ObjectAnnotation& annotation);

  float RotationAngleFromPose(const Matrix3fRM& rotation,
                              const Vector3f& translation, const Vector3f& vec);
  ViewStatus status_;
  float off_threshold_;
  float on_threshold_;
};
REGISTER_CALCULATOR(FrameAnnotationToRectCalculator);

absl::Status FrameAnnotationToRectCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputFrameAnnotationTag)) {
    cc->Inputs().Tag(kInputFrameAnnotationTag).Set<FrameAnnotation>();
  }

  if (cc->Outputs().HasTag(kOutputNormRectsTag)) {
    cc->Outputs().Tag(kOutputNormRectsTag).Set<std::vector<NormalizedRect>>();
  }
  return absl::OkStatus();
}

absl::Status FrameAnnotationToRectCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  status_ = TOP_VIEW_OFF;
  const auto& options = cc->Options<FrameAnnotationToRectCalculatorOptions>();
  off_threshold_ = options.off_threshold();
  on_threshold_ = options.on_threshold();
  RET_CHECK(off_threshold_ <= on_threshold_);
  return absl::OkStatus();
}

absl::Status FrameAnnotationToRectCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().Tag(kInputFrameAnnotationTag).IsEmpty()) {
    return absl::OkStatus();
  }
  auto output_rects = absl::make_unique<std::vector<NormalizedRect>>();
  const auto& frame_annotation =
      cc->Inputs().Tag(kInputFrameAnnotationTag).Get<FrameAnnotation>();
  for (const auto& object_annotation : frame_annotation.annotations()) {
    AddAnnotationToRect(object_annotation, output_rects.get());
  }

  // Output.
  cc->Outputs()
      .Tag(kOutputNormRectsTag)
      .Add(output_rects.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

void FrameAnnotationToRectCalculator::AddAnnotationToRect(
    const ObjectAnnotation& annotation, std::vector<NormalizedRect>* rects) {
  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::min();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::min();
  for (const auto& keypoint : annotation.keypoints()) {
    const auto& point_2d = keypoint.point_2d();
    x_min = std::min(x_min, point_2d.x());
    x_max = std::max(x_max, point_2d.x());
    y_min = std::min(y_min, point_2d.y());
    y_max = std::max(y_max, point_2d.y());
  }
  NormalizedRect new_rect;
  new_rect.set_x_center((x_min + x_max) / 2);
  new_rect.set_y_center((y_min + y_max) / 2);
  new_rect.set_width(x_max - x_min);
  new_rect.set_height(y_max - y_min);
  new_rect.set_rotation(RotationAngleFromAnnotation(annotation));
  rects->push_back(new_rect);
}

float FrameAnnotationToRectCalculator::RotationAngleFromAnnotation(
    const ObjectAnnotation& annotation) {
  // Get box rotation and translation from annotation.
  const auto box_rotation =
      Eigen::Map<const Matrix3fRM>(annotation.rotation().data());
  const auto box_translation =
      Eigen::Map<const Vector3f>(annotation.translation().data());

  // Rotation angle to use when top-view is on(top-view on),
  // Which will make z-axis upright after the rotation.
  const float angle_on =
      RotationAngleFromPose(box_rotation, box_translation, Vector3f::UnitZ());
  // Rotation angle to use when side-view is on(top-view off),
  // Which will make y-axis upright after the rotation.
  const float angle_off =
      RotationAngleFromPose(box_rotation, box_translation, Vector3f::UnitY());

  // Calculate angle between z-axis and viewing ray in degrees.
  const float view_to_z_angle = std::acos(box_rotation(2, 1)) * 180 / M_PI;

  // Determine threshold based on current status,
  // on_threshold_ is used for TOP_VIEW_ON -> TOP_VIEW_OFF transition,
  // off_threshold_ is used for TOP_VIEW_OFF -> TOP_VIEW_ON transition.
  const float thresh =
      (status_ == TOP_VIEW_ON) ? on_threshold_ : off_threshold_;

  // If view_to_z_angle is smaller than threshold, then top-view is on;
  // Otherwise top-view is off.
  status_ = (view_to_z_angle < thresh) ? TOP_VIEW_ON : TOP_VIEW_OFF;

  // Determine which angle to used based on current status_.
  float angle_to_rotate = (status_ == TOP_VIEW_ON) ? angle_on : angle_off;
  return angle_to_rotate;
}

float FrameAnnotationToRectCalculator::RotationAngleFromPose(
    const Matrix3fRM& rotation, const Vector3f& translation,
    const Vector3f& vec) {
  auto p1 = rotation * vec + translation;
  auto p2 = -rotation * vec + translation;
  const float dy = p2[2] * p2[1] - p1[2] * p1[1];
  const float dx = p2[2] * p2[0] - p1[2] * p1[0];
  return M_PI / 2 - std::atan2(dy, dx);
}

}  // namespace mediapipe
