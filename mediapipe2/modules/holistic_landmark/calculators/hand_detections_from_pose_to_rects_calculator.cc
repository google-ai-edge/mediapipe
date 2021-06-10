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

namespace {}  // namespace

// Generates a hand ROI based on a hand detection derived from hand-related pose
// landmarks.
//
// Inputs:
//   DETECTION - Detection.
//     Detection to convert to ROI. Must contain 3 key points indicating: wrist,
//     pinky and index fingers.
//
//   IMAGE_SIZE - std::pair<int, int>
//     Image width and height.
//
// Outputs:
//   NORM_RECT - NormalizedRect.
//     ROI based on passed input.
//
// Examples
// node {
//   calculator: "HandDetectionsFromPoseToRectsCalculator"
//   input_stream: "DETECTION:hand_detection_from_pose"
//   input_stream: "IMAGE_SIZE:image_size"
//   output_stream: "NORM_RECT:hand_roi_from_pose"
// }
class HandDetectionsFromPoseToRectsCalculator
    : public DetectionsToRectsCalculator {
 public:
  absl::Status Open(CalculatorContext* cc) override;

 private:
  ::absl::Status DetectionToNormalizedRect(const Detection& detection,
                                           const DetectionSpec& detection_spec,
                                           NormalizedRect* rect) override;
  absl::Status ComputeRotation(const Detection& detection,
                               const DetectionSpec& detection_spec,
                               float* rotation) override;
};
REGISTER_CALCULATOR(HandDetectionsFromPoseToRectsCalculator);

namespace {

constexpr int kWrist = 0;
constexpr int kPinky = 1;
constexpr int kIndex = 2;

constexpr char kImageSizeTag[] = "IMAGE_SIZE";

}  // namespace

::absl::Status HandDetectionsFromPoseToRectsCalculator::Open(
    CalculatorContext* cc) {
  RET_CHECK(cc->Inputs().HasTag(kImageSizeTag))
      << "Image size is required to calculate rotated rect.";
  cc->SetOffset(TimestampDiff(0));
  target_angle_ = M_PI * 0.5f;
  rotate_ = true;
  options_ = cc->Options<DetectionsToRectsCalculatorOptions>();
  output_zero_rect_for_empty_detections_ =
      options_.output_zero_rect_for_empty_detections();

  return ::absl::OkStatus();
}

::absl::Status
HandDetectionsFromPoseToRectsCalculator ::DetectionToNormalizedRect(
    const Detection& detection, const DetectionSpec& detection_spec,
    NormalizedRect* rect) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  const float x_wrist =
      location_data.relative_keypoints(kWrist).x() * image_size->first;
  const float y_wrist =
      location_data.relative_keypoints(kWrist).y() * image_size->second;

  const float x_index =
      location_data.relative_keypoints(kIndex).x() * image_size->first;
  const float y_index =
      location_data.relative_keypoints(kIndex).y() * image_size->second;

  const float x_pinky =
      location_data.relative_keypoints(kPinky).x() * image_size->first;
  const float y_pinky =
      location_data.relative_keypoints(kPinky).y() * image_size->second;

  // Estimate middle finger.
  const float x_middle = (2.f * x_index + x_pinky) / 3.f;
  const float y_middle = (2.f * y_index + y_pinky) / 3.f;

  // Crop center as middle finger.
  const float center_x = x_middle;
  const float center_y = y_middle;

  // Bounding box size as double distance from middle finger to wrist.
  const float box_size =
      std::sqrt((x_middle - x_wrist) * (x_middle - x_wrist) +
                (y_middle - y_wrist) * (y_middle - y_wrist)) *
      2.0;

  // Set resulting bounding box.
  rect->set_x_center(center_x / image_size->first);
  rect->set_y_center(center_y / image_size->second);
  rect->set_width(box_size / image_size->first);
  rect->set_height(box_size / image_size->second);

  return ::absl::OkStatus();
}

absl::Status HandDetectionsFromPoseToRectsCalculator::ComputeRotation(
    const Detection& detection, const DetectionSpec& detection_spec,
    float* rotation) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate rotation";

  const float x_wrist =
      location_data.relative_keypoints(kWrist).x() * image_size->first;
  const float y_wrist =
      location_data.relative_keypoints(kWrist).y() * image_size->second;

  const float x_index =
      location_data.relative_keypoints(kIndex).x() * image_size->first;
  const float y_index =
      location_data.relative_keypoints(kIndex).y() * image_size->second;

  const float x_pinky =
      location_data.relative_keypoints(kPinky).x() * image_size->first;
  const float y_pinky =
      location_data.relative_keypoints(kPinky).y() * image_size->second;

  // Estimate middle finger.
  const float x_middle = (2.f * x_index + x_pinky) / 3.f;
  const float y_middle = (2.f * y_index + y_pinky) / 3.f;

  *rotation = NormalizeRadians(
      target_angle_ - std::atan2(-(y_middle - y_wrist), x_middle - x_wrist));

  return ::absl::OkStatus();
}

}  // namespace mediapipe
