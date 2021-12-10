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

#include <algorithm>
#include <memory>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/examples/desktop/autoflip/calculators/face_to_region_calculator.pb.h"
#include "mediapipe/examples/desktop/autoflip/quality/visual_scorer.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace mediapipe {
namespace autoflip {

constexpr char kRegionsTag[] = "REGIONS";
constexpr char kFacesTag[] = "FACES";
constexpr char kVideoTag[] = "VIDEO";

// This calculator converts detected faces to SalientRegion protos that can be
// used for downstream processing. Each SalientRegion is scored using image
// cues. Scoring can be controlled through
// FaceToRegionCalculator::scorer_options.
// Example:
//    calculator: "FaceToRegionCalculator"
//    input_stream: "VIDEO:frames"
//    input_stream: "FACES:faces"
//    output_stream: "REGIONS:regions"
//    options:{
//      [mediapipe.autoflip.FaceToRegionCalculatorOptions.ext]:{
//        export_individual_face_landmarks: false
//        export_whole_face: true
//      }
//    }
//
class FaceToRegionCalculator : public CalculatorBase {
 public:
  FaceToRegionCalculator();
  ~FaceToRegionCalculator() override {}
  FaceToRegionCalculator(const FaceToRegionCalculator&) = delete;
  FaceToRegionCalculator& operator=(const FaceToRegionCalculator&) = delete;

  static absl::Status GetContract(mediapipe::CalculatorContract* cc);
  absl::Status Open(mediapipe::CalculatorContext* cc) override;
  absl::Status Process(mediapipe::CalculatorContext* cc) override;

 private:
  double NormalizeX(const int pixel);
  double NormalizeY(const int pixel);
  // Extend the given SalientRegion to include the given point.
  void ExtendSalientRegionWithPoint(const float x, const float y,
                                    SalientRegion* region);
  // Calculator options.
  FaceToRegionCalculatorOptions options_;

  // A scorer used to assign weights to faces.
  std::unique_ptr<VisualScorer> scorer_;
  // Dimensions of video frame
  int frame_width_;
  int frame_height_;
};
REGISTER_CALCULATOR(FaceToRegionCalculator);

FaceToRegionCalculator::FaceToRegionCalculator() {}

absl::Status FaceToRegionCalculator::GetContract(
    mediapipe::CalculatorContract* cc) {
  if (cc->Inputs().HasTag(kVideoTag)) {
    cc->Inputs().Tag(kVideoTag).Set<ImageFrame>();
  }
  cc->Inputs().Tag(kFacesTag).Set<std::vector<mediapipe::Detection>>();
  cc->Outputs().Tag(kRegionsTag).Set<DetectionSet>();
  return absl::OkStatus();
}

absl::Status FaceToRegionCalculator::Open(mediapipe::CalculatorContext* cc) {
  options_ = cc->Options<FaceToRegionCalculatorOptions>();
  if (!cc->Inputs().HasTag(kVideoTag)) {
    RET_CHECK(!options_.use_visual_scorer())
        << "VIDEO input must be provided when using visual_scorer.";
    RET_CHECK(!options_.export_individual_face_landmarks())
        << "VIDEO input must be provided when export_individual_face_landmarks "
           "is set true.";
    RET_CHECK(!options_.export_bbox_from_landmarks())
        << "VIDEO input must be provided when export_bbox_from_landmarks "
           "is set true.";
  }

  scorer_ = absl::make_unique<VisualScorer>(options_.scorer_options());
  frame_width_ = -1;
  frame_height_ = -1;
  return absl::OkStatus();
}

inline double FaceToRegionCalculator::NormalizeX(const int pixel) {
  return pixel / static_cast<double>(frame_width_);
}

inline double FaceToRegionCalculator::NormalizeY(const int pixel) {
  return pixel / static_cast<double>(frame_height_);
}

void FaceToRegionCalculator::ExtendSalientRegionWithPoint(
    const float x, const float y, SalientRegion* region) {
  auto* location = region->mutable_location_normalized();
  if (!location->has_width()) {
    location->set_width(NormalizeX(1));
  } else if (x < location->x()) {
    location->set_width(location->width() + location->x() - x);
  } else if (x > location->x() + location->width()) {
    location->set_width(x - location->x());
  }
  if (!location->has_height()) {
    location->set_height(NormalizeY(1));
  } else if (y < location->y()) {
    location->set_height(location->height() + location->y() - y);
  } else if (y > location->y() + location->height()) {
    location->set_height(y - location->y());
  }

  if (!location->has_x()) {
    location->set_x(x);
  } else {
    location->set_x(std::min(location->x(), x));
  }
  if (!location->has_y()) {
    location->set_y(y);
  } else {
    location->set_y(std::min(location->y(), y));
  }
}

absl::Status FaceToRegionCalculator::Process(mediapipe::CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kVideoTag) &&
      cc->Inputs().Tag(kVideoTag).Value().IsEmpty()) {
    return mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
           << "No VIDEO input at time " << cc->InputTimestamp().Seconds();
  }

  cv::Mat frame;
  if (cc->Inputs().HasTag(kVideoTag)) {
    frame = mediapipe::formats::MatView(
        &cc->Inputs().Tag(kVideoTag).Get<ImageFrame>());
    frame_width_ = frame.cols;
    frame_height_ = frame.rows;
  }

  auto region_set = ::absl::make_unique<DetectionSet>();
  if (!cc->Inputs().Tag(kFacesTag).Value().IsEmpty()) {
    const auto& input_faces =
        cc->Inputs().Tag(kFacesTag).Get<std::vector<mediapipe::Detection>>();

    for (const auto& input_face : input_faces) {
      RET_CHECK(input_face.location_data().format() ==
                mediapipe::LocationData::RELATIVE_BOUNDING_BOX)
          << "Face detection input is lacking required relative_bounding_box()";
      // 6 landmarks should be provided, ordered as:
      // Left eye, Right eye, Nose tip, Mouth center, Left ear tragion, Right
      // ear tragion.
      RET_CHECK(input_face.location_data().relative_keypoints().size() == 6)
          << "Face detection input expected 6 keypoints, has "
          << input_face.location_data().relative_keypoints().size();

      const auto& location = input_face.location_data().relative_bounding_box();

      // Reduce region size to only contain parts of the image in frame.
      float x = std::max(0.0f, location.xmin());
      float y = std::max(0.0f, location.ymin());
      float width =
          std::min(location.width() - abs(x - location.xmin()), 1 - x);
      float height =
          std::min(location.height() - abs(y - location.ymin()), 1 - y);

      // Convert the face to a region.
      if (options_.export_whole_face()) {
        SalientRegion* region = region_set->add_detections();
        region->mutable_location_normalized()->set_x(x);
        region->mutable_location_normalized()->set_y(y);
        region->mutable_location_normalized()->set_width(width);
        region->mutable_location_normalized()->set_height(height);
        region->mutable_signal_type()->set_standard(SignalType::FACE_FULL);

        // Score the face based on image cues.
        float visual_score = 1.0f;
        if (options_.use_visual_scorer()) {
          MP_RETURN_IF_ERROR(
              scorer_->CalculateScore(frame, *region, &visual_score));
        }
        region->set_score(visual_score);
      }

      // Generate two more output regions from important face landmarks. One
      // includes all exterior landmarks, such as ears and chin, and the
      // other includes only interior landmarks, such as the eye edges and the
      // mouth.
      SalientRegion core_landmark_region, all_landmark_region;
      // Keypoints are ordered: Left Eye, Right Eye, Nose Tip, Mouth Center,
      // Left Ear Tragion, Right Ear Tragion.

      // Set 'core' landmarks (Left Eye, Right Eye, Nose Tip, Mouth Center)
      for (int i = 0; i < 4; i++) {
        const auto& keypoint = input_face.location_data().relative_keypoints(i);
        if (options_.export_individual_face_landmarks()) {
          SalientRegion* region = region_set->add_detections();
          region->mutable_location_normalized()->set_x(keypoint.x());
          region->mutable_location_normalized()->set_y(keypoint.y());
          region->mutable_location_normalized()->set_width(NormalizeX(1));
          region->mutable_location_normalized()->set_height(NormalizeY(1));
          region->mutable_signal_type()->set_standard(
              SignalType::FACE_LANDMARK);
        }

        // Extend the core/full landmark regions to include the new
        ExtendSalientRegionWithPoint(keypoint.x(), keypoint.y(),
                                     &core_landmark_region);
        ExtendSalientRegionWithPoint(keypoint.x(), keypoint.y(),
                                     &all_landmark_region);
      }
      // Set 'all' landmarks (Left Ear Tragion, Right Ear Tragion + core)
      for (int i = 4; i < 6; i++) {
        const auto& keypoint = input_face.location_data().relative_keypoints(i);
        if (options_.export_individual_face_landmarks()) {
          SalientRegion* region = region_set->add_detections();
          region->mutable_location()->set_x(keypoint.x());
          region->mutable_location()->set_y(keypoint.y());
          region->mutable_location()->set_width(NormalizeX(1));
          region->mutable_location()->set_height(NormalizeY(1));
          region->mutable_signal_type()->set_standard(
              SignalType::FACE_LANDMARK);
        }

        // Extend the full landmark region to include the new landmark.
        ExtendSalientRegionWithPoint(keypoint.x(), keypoint.y(),
                                     &all_landmark_region);
      }

      // Generate scores for the landmark bboxes and export them.
      if (options_.export_bbox_from_landmarks() &&
          core_landmark_region.has_location_normalized()) {  // Not empty.
        float visual_score = 1.0f;
        if (options_.use_visual_scorer()) {
          MP_RETURN_IF_ERROR(scorer_->CalculateScore(
              frame, core_landmark_region, &visual_score));
        }
        core_landmark_region.set_score(visual_score);
        core_landmark_region.mutable_signal_type()->set_standard(
            SignalType::FACE_CORE_LANDMARKS);
        *region_set->add_detections() = core_landmark_region;
      }
      if (options_.export_bbox_from_landmarks() &&
          all_landmark_region.has_location_normalized()) {  // Not empty.
        float visual_score = 1.0f;
        if (options_.use_visual_scorer()) {
          MP_RETURN_IF_ERROR(scorer_->CalculateScore(frame, all_landmark_region,
                                                     &visual_score));
        }
        all_landmark_region.set_score(visual_score);
        all_landmark_region.mutable_signal_type()->set_standard(
            SignalType::FACE_ALL_LANDMARKS);
        *region_set->add_detections() = all_landmark_region;
      }
    }
  }
  cc->Outputs()
      .Tag(kRegionsTag)
      .Add(region_set.release(), cc->InputTimestamp());

  return absl::OkStatus();
}

}  // namespace autoflip
}  // namespace mediapipe
