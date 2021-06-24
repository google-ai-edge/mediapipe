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
#include "mediapipe/calculators/util/detections_to_rects_calculator.h"

#include <cmath>
#include <limits>

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

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kNormRectsTag[] = "NORM_RECTS";

constexpr float kMinFloat = std::numeric_limits<float>::lowest();
constexpr float kMaxFloat = std::numeric_limits<float>::max();

absl::Status NormRectFromKeyPoints(const LocationData& location_data,
                                   NormalizedRect* rect) {
  RET_CHECK_GT(location_data.relative_keypoints_size(), 1)
      << "2 or more key points required to calculate a rect.";
  float xmin = kMaxFloat;
  float ymin = kMaxFloat;
  float xmax = kMinFloat;
  float ymax = kMinFloat;
  for (int i = 0; i < location_data.relative_keypoints_size(); ++i) {
    const auto& kp = location_data.relative_keypoints(i);
    xmin = std::min(xmin, kp.x());
    ymin = std::min(ymin, kp.y());
    xmax = std::max(xmax, kp.x());
    ymax = std::max(ymax, kp.y());
  }
  rect->set_x_center((xmin + xmax) / 2);
  rect->set_y_center((ymin + ymax) / 2);
  rect->set_width(xmax - xmin);
  rect->set_height(ymax - ymin);
  return absl::OkStatus();
}

template <class B, class R>
void RectFromBox(B box, R* rect) {
  rect->set_x_center(box.xmin() + box.width() / 2);
  rect->set_y_center(box.ymin() + box.height() / 2);
  rect->set_width(box.width());
  rect->set_height(box.height());
}

}  // namespace

absl::Status DetectionsToRectsCalculator::DetectionToRect(
    const Detection& detection, const DetectionSpec& detection_spec,
    Rect* rect) {
  const LocationData location_data = detection.location_data();
  switch (options_.conversion_mode()) {
    case mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode_DEFAULT:
    case mediapipe::
        DetectionsToRectsCalculatorOptions_ConversionMode_USE_BOUNDING_BOX: {
      RET_CHECK(location_data.format() == LocationData::BOUNDING_BOX)
          << "Only Detection with formats of BOUNDING_BOX can be converted to "
             "Rect";
      RectFromBox(location_data.bounding_box(), rect);
      break;
    }
    case mediapipe::
        DetectionsToRectsCalculatorOptions_ConversionMode_USE_KEYPOINTS: {
      RET_CHECK(detection_spec.image_size.has_value())
          << "Rect with absolute coordinates calculation requires image size.";
      const int width = detection_spec.image_size->first;
      const int height = detection_spec.image_size->second;
      NormalizedRect norm_rect;
      MP_RETURN_IF_ERROR(NormRectFromKeyPoints(location_data, &norm_rect));
      rect->set_x_center(std::round(norm_rect.x_center() * width));
      rect->set_y_center(std::round(norm_rect.y_center() * height));
      rect->set_width(std::round(norm_rect.width() * width));
      rect->set_height(std::round(norm_rect.height() * height));
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status DetectionsToRectsCalculator::DetectionToNormalizedRect(
    const Detection& detection, const DetectionSpec& detection_spec,
    NormalizedRect* rect) {
  const LocationData location_data = detection.location_data();
  switch (options_.conversion_mode()) {
    case mediapipe::DetectionsToRectsCalculatorOptions_ConversionMode_DEFAULT:
    case mediapipe::
        DetectionsToRectsCalculatorOptions_ConversionMode_USE_BOUNDING_BOX: {
      RET_CHECK(location_data.format() == LocationData::RELATIVE_BOUNDING_BOX)
          << "Only Detection with formats of RELATIVE_BOUNDING_BOX can be "
             "converted to NormalizedRect";
      RectFromBox(location_data.relative_bounding_box(), rect);
      break;
    }
    case mediapipe::
        DetectionsToRectsCalculatorOptions_ConversionMode_USE_KEYPOINTS: {
      MP_RETURN_IF_ERROR(NormRectFromKeyPoints(location_data, rect));
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status DetectionsToRectsCalculator::GetContract(CalculatorContract* cc) {
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

  return absl::OkStatus();
}

absl::Status DetectionsToRectsCalculator::Open(CalculatorContext* cc) {
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

  return absl::OkStatus();
}

absl::Status DetectionsToRectsCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kDetectionTag) &&
      cc->Inputs().Tag(kDetectionTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().HasTag(kDetectionsTag) &&
      cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (rotate_ && !HasTagValue(cc, kImageSizeTag)) {
    return absl::OkStatus();
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
        if (cc->Outputs().HasTag(kNormRectsTag)) {
          auto rect_vector = absl::make_unique<std::vector<NormalizedRect>>();
          rect_vector->emplace_back(NormalizedRect());
          cc->Outputs()
              .Tag(kNormRectsTag)
              .Add(rect_vector.release(), cc->InputTimestamp());
        }
      }
      return absl::OkStatus();
    }
  }

  // Get dynamic calculator options (e.g. `image_size`).
  const DetectionSpec detection_spec = GetDetectionSpec(cc);

  if (cc->Outputs().HasTag(kRectTag)) {
    auto output_rect = absl::make_unique<Rect>();
    MP_RETURN_IF_ERROR(
        DetectionToRect(detections[0], detection_spec, output_rect.get()));
    if (rotate_) {
      float rotation;
      MP_RETURN_IF_ERROR(
          ComputeRotation(detections[0], detection_spec, &rotation));
      output_rect->set_rotation(rotation);
    }
    cc->Outputs().Tag(kRectTag).Add(output_rect.release(),
                                    cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kNormRectTag)) {
    auto output_rect = absl::make_unique<NormalizedRect>();
    MP_RETURN_IF_ERROR(DetectionToNormalizedRect(detections[0], detection_spec,
                                                 output_rect.get()));
    if (rotate_) {
      float rotation;
      MP_RETURN_IF_ERROR(
          ComputeRotation(detections[0], detection_spec, &rotation));
      output_rect->set_rotation(rotation);
    }
    cc->Outputs()
        .Tag(kNormRectTag)
        .Add(output_rect.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kRectsTag)) {
    auto output_rects = absl::make_unique<std::vector<Rect>>(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
      MP_RETURN_IF_ERROR(DetectionToRect(detections[i], detection_spec,
                                         &(output_rects->at(i))));
      if (rotate_) {
        float rotation;
        MP_RETURN_IF_ERROR(
            ComputeRotation(detections[i], detection_spec, &rotation));
        output_rects->at(i).set_rotation(rotation);
      }
    }
    cc->Outputs().Tag(kRectsTag).Add(output_rects.release(),
                                     cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kNormRectsTag)) {
    auto output_rects =
        absl::make_unique<std::vector<NormalizedRect>>(detections.size());
    for (int i = 0; i < detections.size(); ++i) {
      MP_RETURN_IF_ERROR(DetectionToNormalizedRect(
          detections[i], detection_spec, &(output_rects->at(i))));
      if (rotate_) {
        float rotation;
        MP_RETURN_IF_ERROR(
            ComputeRotation(detections[i], detection_spec, &rotation));
        output_rects->at(i).set_rotation(rotation);
      }
    }
    cc->Outputs()
        .Tag(kNormRectsTag)
        .Add(output_rects.release(), cc->InputTimestamp());
  }

  return absl::OkStatus();
}

absl::Status DetectionsToRectsCalculator::ComputeRotation(
    const Detection& detection, const DetectionSpec& detection_spec,
    float* rotation) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  const float x0 = location_data.relative_keypoints(start_keypoint_index_).x() *
                   image_size->first;
  const float y0 = location_data.relative_keypoints(start_keypoint_index_).y() *
                   image_size->second;
  const float x1 = location_data.relative_keypoints(end_keypoint_index_).x() *
                   image_size->first;
  const float y1 = location_data.relative_keypoints(end_keypoint_index_).y() *
                   image_size->second;

  *rotation = NormalizeRadians(target_angle_ - std::atan2(-(y1 - y0), x1 - x0));

  return absl::OkStatus();
}

DetectionSpec DetectionsToRectsCalculator::GetDetectionSpec(
    const CalculatorContext* cc) {
  absl::optional<std::pair<int, int>> image_size;
  if (HasTagValue(cc->Inputs(), kImageSizeTag)) {
    image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();
  }

  return {image_size};
}

REGISTER_CALCULATOR(DetectionsToRectsCalculator);

}  // namespace mediapipe
