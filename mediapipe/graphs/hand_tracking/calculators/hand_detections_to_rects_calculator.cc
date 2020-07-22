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
// limitations under the License.

#include <cmath>

#include "mediapipe/calculators/util/detections_to_rects_calculator.h"
#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace {

// Indices of joints used for computing the rotation of the output rectangle
// from detection box with keypoints.
constexpr int kWristJoint = 0;
constexpr int kMiddleFingerPIPJoint = 6;
constexpr int kIndexFingerPIPJoint = 4;
constexpr int kRingFingerPIPJoint = 8;
constexpr char kImageSizeTag[] = "IMAGE_SIZE";

}  // namespace

// A calculator that converts Hand detection to a bounding box NormalizedRect.
// The calculator overwrites the default logic of DetectionsToRectsCalculator
// for rotating the detection bounding to a rectangle. The rotation angle is
// computed based on 1) the wrist joint and 2) the average of PIP joints of
// index finger, middle finger and ring finger. After rotation, the vector from
// the wrist to the mean of PIP joints is expected to be vertical with wrist at
// the bottom and the mean of PIP joints at the top.
class HandDetectionsToRectsCalculator : public DetectionsToRectsCalculator {
 public:
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag))
        << "Image size is required to calculate rotated rect";
    cc->SetOffset(TimestampDiff(0));
    target_angle_ = M_PI * 0.5f;
    rotate_ = true;
    options_ = cc->Options<DetectionsToRectsCalculatorOptions>();
    output_zero_rect_for_empty_detections_ =
        options_.output_zero_rect_for_empty_detections();

    return ::mediapipe::OkStatus();
  }

 private:
  ::mediapipe::Status ComputeRotation(const ::mediapipe::Detection& detection,
                                      const DetectionSpec& detection_spec,
                                      float* rotation) override;
};
REGISTER_CALCULATOR(HandDetectionsToRectsCalculator);

::mediapipe::Status HandDetectionsToRectsCalculator::ComputeRotation(
    const Detection& detection, const DetectionSpec& detection_spec,
    float* rotation) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  const float x0 =
      location_data.relative_keypoints(kWristJoint).x() * image_size->first;
  const float y0 =
      location_data.relative_keypoints(kWristJoint).y() * image_size->second;

  float x1 = (location_data.relative_keypoints(kIndexFingerPIPJoint).x() +
              location_data.relative_keypoints(kRingFingerPIPJoint).x()) /
             2.f;
  float y1 = (location_data.relative_keypoints(kIndexFingerPIPJoint).y() +
              location_data.relative_keypoints(kRingFingerPIPJoint).y()) /
             2.f;
  x1 = (x1 + location_data.relative_keypoints(kMiddleFingerPIPJoint).x()) /
       2.f * image_size->first;
  y1 = (y1 + location_data.relative_keypoints(kMiddleFingerPIPJoint).y()) /
       2.f * image_size->second;

  *rotation = NormalizeRadians(target_angle_ - std::atan2(-(y1 - y0), x1 - x0));

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
