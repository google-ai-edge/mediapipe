// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/calculators/util/detections_to_rects_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

using mediapipe::DetectionsToRectsCalculatorOptions;

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kNormRectsTag[] = "NORM_RECTS";

::mediapipe::Status DetectionToRect(const Detection& detection, Rect* rect) {
  const LocationData location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::BOUNDING_BOX)
      << "Only Detection with formats of BOUNDING_BOX can be converted to Rect";
  const LocationData::BoundingBox bounding_box = location_data.bounding_box();
  rect->set_x_center(bounding_box.xmin() + bounding_box.width() / 2);
  rect->set_y_center(bounding_box.ymin() + bounding_box.height() / 2);
  rect->set_width(bounding_box.width());
  rect->set_height(bounding_box.height());
  return ::mediapipe::OkStatus();
}

::mediapipe::Status DetectionToNormalizedRect(const Detection& detection,
                                              NormalizedRect* rect) {
  const LocationData location_data = detection.location_data();
  RET_CHECK(location_data.format() == LocationData::RELATIVE_BOUNDING_BOX)
      << "Only Detection with formats of RELATIVE_BOUNDING_BOX can be "
         "converted to NormalizedRect";
  const LocationData::RelativeBoundingBox bounding_box =
      location_data.relative_bounding_box();
  rect->set_x_center(bounding_box.xmin() + bounding_box.width() / 2);
  rect->set_y_center(bounding_box.ymin() + bounding_box.height() / 2);
  rect->set_width(bounding_box.width());
  rect->set_height(bounding_box.height());
  return ::mediapipe::OkStatus();
}

// Wraps around an angle in radians to within -M_PI and M_PI.
inline float NormalizeRadians(float angle) {
  return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

}  // namespace

// A calculator that converts Detection proto to Rect proto.
//
// Detection is the format for encoding one or more detections in an image.
// The input can be a single Detection or std::vector<Detection>. The output can
// be either a single Rect or NormalizedRect, or std::vector<Rect> or
// std::vector<NormalizedRect>. If Rect is used, the LocationData format is
// expected to be BOUNDING_BOX, and if NormalizedRect is used it is expected to
// be RELATIVE_BOUNDING_BOX.
//
// When the input is std::vector<Detection> and the output is a Rect or
// NormalizedRect, only the first detection is converted. When the input is a
// single Detection and the output is a std::vector<Rect> or
// std::vector<NormalizedRect>, the output is a vector of size 1.
//
// Inputs:
//
// One of the following:
// DETECTION: A Detection proto.
// DETECTIONS: An std::vector<Detection>.
//
// IMAGE_SIZE (optional): A std::pair<int, int> represention image width and
//   height. This is required only when rotation needs to be computed (see
//   calculator options).
//
// Output:
// One of the following:
// RECT: A Rect proto.
// NORM_RECT: A NormalizedRect proto.
// RECTS: An std::vector<Rect>.
// NORM_RECTS: An std::vector<NormalizedRect>.
//
// Example config:
// node {
//   calculator: "DetectionsToRectsCalculator"
//   input_stream: "DETECTIONS:detections"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "NORM_RECT:rect"
//   options: {
//     [mediapipe.DetectionsToRectCalculatorOptions.ext] {
//       rotation_vector_start_keypoint_index: 0
//       rotation_vector_end_keypoint_index: 2
//       rotation_vector_target_angle_degrees: 90
//       output_zero_rect_for_empty_detections: true
//     }
//   }
// }
class DetectionsToRectsCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  float ComputeRotation(const Detection& detection,
                        const std::pair<int, int> image_size);

  DetectionsToRectsCalculatorOptions options_;
  int start_keypoint_index_;
  int end_keypoint_index_;
  float target_angle_;  // In radians.
  bool rotate_;
  bool output_zero_rect_for_empty_detections_;
};
REGISTER_CALCULATOR(DetectionsToRectsCalculator);

::mediapipe::Status DetectionsToRectsCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kDetectionTag) ^
            cc->Inputs().HasTag(kDetectionsTag))
      << "Exactly one of DETECTION or DETECTIONS input stream should be "
         "provided.";
  RET_CHECK_EQ((cc->Outputs().HasTag(kNormRectTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kRectTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kNormRectsTag) ? 1 : 0) +
                   (cc->Outputs().HasTag(kRectsTag) ? 1 : 0),
               1)
      << "Exactly one of NORM_RECT, RECT, NORM_RECTS or RECTS output stream "
         "should be provided.";

  if (cc->Inputs().HasTag(kDetectionTag)) {
    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  }
  if (cc->Inputs().HasTag(kImageSizeTag)) {
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  }

  if (cc->Outputs().HasTag(kRectTag)) {
    cc->Outputs().Tag(kRectTag).Set<Rect>();
  }
  if (cc->Outputs().HasTag(kNormRectTag)) {
    cc->Outputs().Tag(kNormRectTag).Set<NormalizedRect>();
  }
  if (cc->Outputs().HasTag(kRectsTag)) {
    cc->Outputs().Tag(kRectsTag).Set<std::vector<Rect>>();
  }
  if (cc->Outputs().HasTag(kNormRectsTag)) {
    cc->Outputs().Tag(kNormRectsTag).Set<std::vector<NormalizedRect>>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status DetectionsToRectsCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<DetectionsToRectsCalculatorOptions>();

  if (options_.has_rotation_vector_start_keypoint_index()) {
    RET_CHECK(options_.has_rotation_vector_end_keypoint_index());
    RET_CHECK(options_.has_rotation_vector_target_angle() ^
              options_.has_rotation_vector_target_angle_degrees());
    RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));

    if (options_.has_rotation_vector_target_angle()) {
      target_angle_ = options_.rotation_vector_target_angle();
    } else {
      target_angle_ =
          M_PI * options_.rotation_vector_target_angle_degrees() / 180.f;
    }
    start_keypoint_index_ = options_.rotation_vector_start_keypoint_index();
    end_keypoint_index_ = options_.rotation_vector_end_keypoint_index();
    rotate_ = true;
  }

  output_zero_rect_for_empty_detections_ =
      options_.output_zero_rect_for_empty_detections();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status DetectionsToRectsCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kDetectionTag) &&
      cc->Inputs().Tag(kDetectionTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }
  if (cc->Inputs().HasTag(kDetectionsTag) &&
      cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    return ::mediapipe::OkStatus();
  }

  std::vector<Detection> detections;
  if (cc->Inputs().HasTag(kDetectionTag)) {
    detections.push_back(cc->Inputs().Tag(kDetectionTag).Get<Detection>());
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    detections = cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    if (detections.empty()) {
      if (output_zero_rect_for_empty_detections_) {
        if (cc->Outputs().HasTag(kRectTag)) {
          cc->Outputs().Tag(kRectTag).AddPacket(
              MakePacket<Rect>().At(cc->InputTimestamp()));
        }
        if (cc->Outputs().HasTag(kNormRectTag)) {
          cc->Outputs()
              .Tag(kNormRectTag)
              .AddPacket(MakePacket<NormalizedRect>().At(cc->InputTimestamp()));
        }
      }
      return ::mediapipe::OkStatus();
    }
  }

  std::pair<int, int> image_size;
  if (rotate_) {
    RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());
    image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
  }

  if (cc->Outputs().HasTag(kRectTag)) {
    auto output_rect = absl::make_unique<Rect>();
    MP_RETURN_IF_ERROR(DetectionToRect(detections[0], output_rect.get()));
    if (rotate_) {
      output_rect->set_rotation(ComputeRotation(detections[0], image_size));
    }
    cc->Outputs().Tag(kRectTag).Add(output_rect.release(),
                                    cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kNormRectTag)) {
    auto output_rect = absl::make_unique<NormalizedRect>();
    MP_RETURN_IF_ERROR(
        DetectionToNormalizedRect(detections[0], output_rect.get()));
    if (rotate_) {
      output_rect->set_rotation(ComputeRotation(detections[0], image_size));
    }
    cc->Outputs()
        .Tag(kNormRectTag)
        .Add(output_rect.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kRectsTag)) {
    auto output_rects = absl::make_unique<std::vector<Rect>>(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
      MP_RETURN_IF_ERROR(
          DetectionToRect(detections[i], &(output_rects->at(i))));
      if (rotate_) {
        output_rects->at(i).set_rotation(
            ComputeRotation(detections[i], image_size));
      }
    }
    cc->Outputs().Tag(kRectsTag).Add(output_rects.release(),
                                     cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kNormRectsTag)) {
    auto output_rects =
        absl::make_unique<std::vector<NormalizedRect>>(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
      MP_RETURN_IF_ERROR(
          DetectionToNormalizedRect(detections[i], &(output_rects->at(i))));
      if (rotate_) {
        output_rects->at(i).set_rotation(
            ComputeRotation(detections[i], image_size));
      }
    }
    cc->Outputs()
        .Tag(kNormRectsTag)
        .Add(output_rects.release(), cc->InputTimestamp());
  }

  return ::mediapipe::OkStatus();
}

float DetectionsToRectsCalculator::ComputeRotation(
    const Detection& detection, const std::pair<int, int> image_size) {
  const auto& location_data = detection.location_data();
  const float x0 = location_data.relative_keypoints(start_keypoint_index_).x() *
                   image_size.first;
  const float y0 = location_data.relative_keypoints(start_keypoint_index_).y() *
                   image_size.second;
  const float x1 = location_data.relative_keypoints(end_keypoint_index_).x() *
                   image_size.first;
  const float y1 = location_data.relative_keypoints(end_keypoint_index_).y() *
                   image_size.second;

  float rotation = target_angle_ - std::atan2(-(y1 - y0), x1 - x0);

  return NormalizeRadians(rotation);
}

}  // namespace mediapipe
