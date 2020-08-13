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

// A calculator that converts Detection with two alignment points to Rect.
//
// Detection should contain two points:
//   * Center point - center of the crop
//   * Scale point - vector from center to scale point defines size and rotation
//       of the Rect. Not that Y coordinate of this vector is flipped before
//       computing the rotation (it is caused by the fact that Y axis is
//       directed downwards). So define target rotation vector accordingly.
//
// Example config:
//   node {
//     calculator: "AlignmentPointsRectsCalculator"
//     input_stream: "DETECTIONS:detections"
//     input_stream: "IMAGE_SIZE:image_size"
//     output_stream: "NORM_RECT:rect"
//     options: {
//       [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
//         rotation_vector_start_keypoint_index: 0
//         rotation_vector_end_keypoint_index: 1
//         rotation_vector_target_angle_degrees: 90
//         output_zero_rect_for_empty_detections: true
//       }
//     }
//   }
class AlignmentPointsRectsCalculator : public DetectionsToRectsCalculator {
 public:
  ::mediapipe::Status Open(CalculatorContext* cc) override {
    RET_CHECK_OK(DetectionsToRectsCalculator::Open(cc));

    // Make sure that start and end keypoints are provided.
    // They are required for the rect size calculation and will also force base
    // calculator to compute rotation.
    options_ = cc->Options<DetectionsToRectsCalculatorOptions>();
    RET_CHECK(options_.has_rotation_vector_start_keypoint_index())
        << "Start keypoint is required to calculate rect size and rotation";
    RET_CHECK(options_.has_rotation_vector_end_keypoint_index())
        << "End keypoint is required to calculate rect size and rotation";

    return ::mediapipe::OkStatus();
  }

 private:
  ::mediapipe::Status DetectionToNormalizedRect(
      const ::mediapipe::Detection& detection,
      const DetectionSpec& detection_spec,
      ::mediapipe::NormalizedRect* rect) override;
};
REGISTER_CALCULATOR(AlignmentPointsRectsCalculator);

::mediapipe::Status AlignmentPointsRectsCalculator::DetectionToNormalizedRect(
    const Detection& detection, const DetectionSpec& detection_spec,
    NormalizedRect* rect) {
  const auto& location_data = detection.location_data();
  const auto& image_size = detection_spec.image_size;
  RET_CHECK(image_size) << "Image size is required to calculate the rect";

  const float x_center =
      location_data.relative_keypoints(start_keypoint_index_).x() *
      image_size->first;
  const float y_center =
      location_data.relative_keypoints(start_keypoint_index_).y() *
      image_size->second;

  const float x_scale =
      location_data.relative_keypoints(end_keypoint_index_).x() *
      image_size->first;
  const float y_scale =
      location_data.relative_keypoints(end_keypoint_index_).y() *
      image_size->second;

  // Bounding box size as double distance from center to scale point.
  const float box_size =
      std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                (y_scale - y_center) * (y_scale - y_center)) *
      2.0;

  // Set resulting bounding box.
  rect->set_x_center(x_center / image_size->first);
  rect->set_y_center(y_center / image_size->second);
  rect->set_width(box_size / image_size->first);
  rect->set_height(box_size / image_size->second);

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
