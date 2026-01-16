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

#include <math.h>

#include <cstdlib>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/rectangle.h"
#include "mediapipe/modules/holistic_landmark/calculators/roi_tracking_calculator.pb.h"

namespace mediapipe {

namespace {

constexpr char kPrevLandmarksTag[] = "PREV_LANDMARKS";
constexpr char kPrevLandmarksRectTag[] = "PREV_LANDMARKS_RECT";
constexpr char kRecropRectTag[] = "RECROP_RECT";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kTrackingRectTag[] = "TRACKING_RECT";

using ::mediapipe::NormalizedRect;

// TODO: Use rect rotation.
// Verifies that Intersection over Union of previous frame rect and current
// frame re-crop rect is less than threshold.
bool IouRequirementsSatisfied(const NormalizedRect& prev_rect,
                              const NormalizedRect& recrop_rect,
                              const std::pair<int, int>& image_size,
                              const float min_iou) {
  auto r1 = Rectangle_f(prev_rect.x_center() * image_size.first,
                        prev_rect.y_center() * image_size.second,
                        prev_rect.width() * image_size.first,
                        prev_rect.height() * image_size.second);
  auto r2 = Rectangle_f(recrop_rect.x_center() * image_size.first,
                        recrop_rect.y_center() * image_size.second,
                        recrop_rect.width() * image_size.first,
                        recrop_rect.height() * image_size.second);

  const float intersection_area = r1.Intersect(r2).Area();
  const float union_area = r1.Area() + r2.Area() - intersection_area;

  const float intersection_threshold = union_area * min_iou;
  if (intersection_area < intersection_threshold) {
    VLOG(1) << absl::StrFormat("Lost tracking: IoU intersection %f < %f",
                               intersection_area, intersection_threshold);
    return false;
  }

  return true;
}

// Verifies that current frame re-crop rect rotation/translation/scale didn't
// change much comparing to the previous frame rect. Translation and scale are
// normalized by current frame re-crop rect.
bool RectRequirementsSatisfied(const NormalizedRect& prev_rect,
                               const NormalizedRect& recrop_rect,
                               const std::pair<int, int> image_size,
                               const float rotation_degrees,
                               const float translation, const float scale) {
  // Rotate both rects so that re-crop rect edges are parallel to XY axes. That
  // will allow to compute x/y translation of the previous frame rect along axes
  // of the current frame re-crop rect.
  const float rotation = -recrop_rect.rotation();

  const float cosa = cos(rotation);
  const float sina = sin(rotation);

  // Rotate previous frame rect and get its parameters.
  const float prev_rect_x = prev_rect.x_center() * image_size.first * cosa -
                            prev_rect.y_center() * image_size.second * sina;
  const float prev_rect_y = prev_rect.x_center() * image_size.first * sina +
                            prev_rect.y_center() * image_size.second * cosa;
  const float prev_rect_width = prev_rect.width() * image_size.first;
  const float prev_rect_height = prev_rect.height() * image_size.second;
  const float prev_rect_rotation = prev_rect.rotation() / M_PI * 180.f;

  // Rotate current frame re-crop rect and get its parameters.
  const float recrop_rect_x = recrop_rect.x_center() * image_size.first * cosa -
                              recrop_rect.y_center() * image_size.second * sina;
  const float recrop_rect_y = recrop_rect.x_center() * image_size.first * sina +
                              recrop_rect.y_center() * image_size.second * cosa;
  const float recrop_rect_width = recrop_rect.width() * image_size.first;
  const float recrop_rect_height = recrop_rect.height() * image_size.second;
  const float recrop_rect_rotation = recrop_rect.rotation() / M_PI * 180.f;

  // Rect requirements are satisfied unless one of the checks below fails.
  bool satisfied = true;

  // Ensure that rotation diff is in [0, 180] range.
  float rotation_diff = prev_rect_rotation - recrop_rect_rotation;
  if (rotation_diff > 180.f) {
    rotation_diff -= 360.f;
  }
  if (rotation_diff < -180.f) {
    rotation_diff += 360.f;
  }
  rotation_diff = abs(rotation_diff);
  if (rotation_diff > rotation_degrees) {
    satisfied = false;
    VLOG(1) << absl::StrFormat("Lost tracking: rect rotation %f > %f",
                               rotation_diff, rotation_degrees);
  }

  const float x_diff = abs(prev_rect_x - recrop_rect_x);
  const float x_threshold = recrop_rect_width * translation;
  if (x_diff > x_threshold) {
    satisfied = false;
    VLOG(1) << absl::StrFormat("Lost tracking: rect x translation %f > %f",
                               x_diff, x_threshold);
  }

  const float y_diff = abs(prev_rect_y - recrop_rect_y);
  const float y_threshold = recrop_rect_height * translation;
  if (y_diff > y_threshold) {
    satisfied = false;
    VLOG(1) << absl::StrFormat("Lost tracking: rect y translation %f > %f",
                               y_diff, y_threshold);
  }

  const float width_diff = abs(prev_rect_width - recrop_rect_width);
  const float width_threshold = recrop_rect_width * scale;
  if (width_diff > width_threshold) {
    satisfied = false;
    VLOG(1) << absl::StrFormat("Lost tracking: rect width %f > %f", width_diff,
                               width_threshold);
  }

  const float height_diff = abs(prev_rect_height - recrop_rect_height);
  const float height_threshold = recrop_rect_height * scale;
  if (height_diff > height_threshold) {
    satisfied = false;
    VLOG(1) << absl::StrFormat("Lost tracking: rect height %f > %f",
                               height_diff, height_threshold);
  }

  return satisfied;
}

// Verifies that landmarks from the previous frame are within re-crop rectangle
// bounds on the current frame.
bool LandmarksRequirementsSatisfied(const NormalizedLandmarkList& landmarks,
                                    const NormalizedRect& recrop_rect,
                                    const std::pair<int, int> image_size,
                                    const float recrop_rect_margin) {
  // Rotate both re-crop rectangle and landmarks so that re-crop rectangle edges
  // are parallel to XY axes. It will allow to easily check if landmarks are
  // within re-crop rect bounds along re-crop rect axes.
  //
  // Rect rotation is specified clockwise. To apply cos/sin functions we
  // transform it into counterclockwise.
  const float rotation = -recrop_rect.rotation();

  const float cosa = cos(rotation);
  const float sina = sin(rotation);

  // Rotate rect.
  const float rect_x = recrop_rect.x_center() * image_size.first * cosa -
                       recrop_rect.y_center() * image_size.second * sina;
  const float rect_y = recrop_rect.x_center() * image_size.first * sina +
                       recrop_rect.y_center() * image_size.second * cosa;
  const float rect_width =
      recrop_rect.width() * image_size.first * (1.f + recrop_rect_margin);
  const float rect_height =
      recrop_rect.height() * image_size.second * (1.f + recrop_rect_margin);

  // Get rect bounds.
  const float rect_left = rect_x - rect_width * 0.5f;
  const float rect_right = rect_x + rect_width * 0.5f;
  const float rect_top = rect_y - rect_height * 0.5f;
  const float rect_bottom = rect_y + rect_height * 0.5f;

  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const auto& landmark = landmarks.landmark(i);
    const float x = landmark.x() * image_size.first * cosa -
                    landmark.y() * image_size.second * sina;
    const float y = landmark.x() * image_size.first * sina +
                    landmark.y() * image_size.second * cosa;

    if (!(rect_left < x && x < rect_right && rect_top < y && y < rect_bottom)) {
      VLOG(1) << "Lost tracking: landmarks out of re-crop rect";
      return false;
    }
  }

  return true;
}

}  // namespace

// A calculator to track object rectangle between frames.
//
// Calculator checks that all requirements for tracking are satisfied and uses
// rectangle from the previous frame in this case, otherwise - uses current
// frame re-crop rectangle.
//
// There are several types of tracking requirements that can be configured via
// options:
//   IoU: Verifies that IoU of the previous frame rectangle and current frame
//     re-crop rectangle is less than a given threshold.
//   Rect parameters: Verifies that rotation/translation/scale of the re-crop
//     rectangle on the current frame is close to the rectangle from the
//     previous frame within given thresholds.
//   Landmarks: Verifies that landmarks from the previous frame are within
//     the re-crop rectangle on the current frame.
//
// Inputs:
//   PREV_LANDMARKS: Object landmarks from the previous frame.
//   PREV_LANDMARKS_RECT: Object rectangle based on the landmarks from the
//     previous frame.
//   RECROP_RECT: Object re-crop rectangle from the current frame.
//   IMAGE_SIZE: Image size to transform normalized coordinates to absolute.
//
// Outputs:
//   TRACKING_RECT: Rectangle to use for object prediction on the current frame.
//     It will be either object rectangle from the previous frame (if all
//     tracking requirements are satisfied) or re-crop rectangle from the
//     current frame (if tracking lost the object).
//
// Example config:
//   node {
//     calculator: "RoiTrackingCalculator"
//     input_stream: "PREV_LANDMARKS:prev_hand_landmarks"
//     input_stream: "PREV_LANDMARKS_RECT:prev_hand_landmarks_rect"
//     input_stream: "RECROP_RECT:hand_recrop_rect"
//     input_stream: "IMAGE_SIZE:image_size"
//     output_stream: "TRACKING_RECT:hand_tracking_rect"
//     options: {
//       [mediapipe.RoiTrackingCalculatorOptions.ext] {
//         rect_requirements: {
//           rotation_degrees: 40.0
//           translation: 0.2
//           scale: 0.4
//         }
//         landmarks_requirements: {
//           recrop_rect_margin: -0.1
//         }
//       }
//     }
//   }
class RoiTrackingCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  RoiTrackingCalculatorOptions options_;
};
REGISTER_CALCULATOR(RoiTrackingCalculator);

absl::Status RoiTrackingCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kPrevLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kPrevLandmarksRectTag).Set<NormalizedRect>();
  cc->Inputs().Tag(kRecropRectTag).Set<NormalizedRect>();
  cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
  cc->Outputs().Tag(kTrackingRectTag).Set<NormalizedRect>();

  return absl::OkStatus();
}

absl::Status RoiTrackingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<RoiTrackingCalculatorOptions>();
  return absl::OkStatus();
}

absl::Status RoiTrackingCalculator::Process(CalculatorContext* cc) {
  // If there is no current frame re-crop rect (i.e. object is not present on
  // the current frame) - return empty packet.
  if (cc->Inputs().Tag(kRecropRectTag).IsEmpty()) {
    return absl::OkStatus();
  }

  // If there is no previous rect, but there is current re-crop rect - return
  // current re-crop rect as is.
  if (cc->Inputs().Tag(kPrevLandmarksRectTag).IsEmpty()) {
    cc->Outputs()
        .Tag(kTrackingRectTag)
        .AddPacket(cc->Inputs().Tag(kRecropRectTag).Value());
    return absl::OkStatus();
  }

  // At this point we have both previous rect (which also means we have previous
  // landmarks) and currrent re-crop rect.
  const auto& prev_landmarks =
      cc->Inputs().Tag(kPrevLandmarksTag).Get<NormalizedLandmarkList>();
  const auto& prev_rect =
      cc->Inputs().Tag(kPrevLandmarksRectTag).Get<NormalizedRect>();
  const auto& recrop_rect =
      cc->Inputs().Tag(kRecropRectTag).Get<NormalizedRect>();
  const auto& image_size =
      cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  // Keep tracking unless one of the requirements below is not satisfied.
  bool keep_tracking = true;

  // If IoU of the previous rect and current re-crop rect is lower than allowed
  // threshold - use current re-crop rect.
  if (options_.has_iou_requirements() &&
      !IouRequirementsSatisfied(prev_rect, recrop_rect, image_size,
                                options_.iou_requirements().min_iou())) {
    keep_tracking = false;
  }

  // If previous rect and current re-crop rect differ more than it is allowed by
  // the augmentations (used during the model training) - use current re-crop
  // rect.
  if (options_.has_rect_requirements() &&
      !RectRequirementsSatisfied(
          prev_rect, recrop_rect, image_size,
          options_.rect_requirements().rotation_degrees(),
          options_.rect_requirements().translation(),
          options_.rect_requirements().scale())) {
    keep_tracking = false;
  }

  // If landmarks from the previous frame are not in the current re-crop rect
  // (i.e. object moved too fast and using previous frame rect won't cover
  // landmarks on the current frame) - use current re-crop rect.
  if (options_.has_landmarks_requirements() &&
      !LandmarksRequirementsSatisfied(
          prev_landmarks, recrop_rect, image_size,
          options_.landmarks_requirements().recrop_rect_margin())) {
    keep_tracking = false;
  }

  // If object didn't move a lot comparing to the previous frame - we'll keep
  // tracking it and will return rect from the previous frame, otherwise -
  // return re-crop rect from the current frame.
  if (keep_tracking) {
    cc->Outputs()
        .Tag(kTrackingRectTag)
        .AddPacket(cc->Inputs().Tag(kPrevLandmarksRectTag).Value());
  } else {
    cc->Outputs()
        .Tag(kTrackingRectTag)
        .AddPacket(cc->Inputs().Tag(kRecropRectTag).Value());
    VLOG(1) << "Lost tracking: check messages above for details";
  }

  return absl::OkStatus();
}

}  // namespace mediapipe
