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

#include <memory>

#include "mediapipe/calculators/util/landmarks_to_detection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kNormalizedLandmarksTag[] = "NORM_LANDMARKS";

Detection ConvertLandmarksToDetection(const NormalizedLandmarkList& landmarks) {
  Detection detection;
  LocationData* location_data = detection.mutable_location_data();

  float x_min = std::numeric_limits<float>::max();
  float x_max = std::numeric_limits<float>::min();
  float y_min = std::numeric_limits<float>::max();
  float y_max = std::numeric_limits<float>::min();
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    const NormalizedLandmark& landmark = landmarks.landmark(i);
    x_min = std::min(x_min, landmark.x());
    x_max = std::max(x_max, landmark.x());
    y_min = std::min(y_min, landmark.y());
    y_max = std::max(y_max, landmark.y());

    auto keypoint = location_data->add_relative_keypoints();
    keypoint->set_x(landmark.x());
    keypoint->set_y(landmark.y());
  }

  location_data->set_format(LocationData::RELATIVE_BOUNDING_BOX);
  LocationData::RelativeBoundingBox* relative_bbox =
      location_data->mutable_relative_bounding_box();

  relative_bbox->set_xmin(x_min);
  relative_bbox->set_ymin(y_min);
  relative_bbox->set_width(x_max - x_min);
  relative_bbox->set_height(y_max - y_min);

  return detection;
}

}  // namespace

// Converts NormalizedLandmark to Detection proto. A relative bounding box will
// be created containing all landmarks exactly. A calculator option is provided
// to specify a subset of landmarks for creating the detection.
//
// Input:
//  NOMR_LANDMARKS: A NormalizedLandmarkList proto.
//
// Output:
//   DETECTION: A Detection proto.
//
// Example config:
// node {
//   calculator: "LandmarksToDetectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   output_stream: "DETECTION:detections"
// }
class LandmarksToDetectionCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  ::mediapipe::LandmarksToDetectionCalculatorOptions options_;
};
REGISTER_CALCULATOR(LandmarksToDetectionCalculator);

::mediapipe::Status LandmarksToDetectionCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kNormalizedLandmarksTag));
  RET_CHECK(cc->Outputs().HasTag(kDetectionTag));
  // TODO: Also support converting Landmark to Detection.
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Outputs().Tag(kDetectionTag).Set<Detection>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToDetectionCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<::mediapipe::LandmarksToDetectionCalculatorOptions>();
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToDetectionCalculator::Process(
    CalculatorContext* cc) {
  const auto& landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_GT(landmarks.landmark_size(), 0)
      << "Input landmark vector is empty.";

  auto detection = absl::make_unique<Detection>();
  if (options_.selected_landmark_indices_size()) {
    NormalizedLandmarkList subset_landmarks;
    for (int i = 0; i < options_.selected_landmark_indices_size(); ++i) {
      RET_CHECK_LT(options_.selected_landmark_indices(i),
                   landmarks.landmark_size())
          << "Index of landmark subset is out of range.";
      *subset_landmarks.add_landmark() =
          landmarks.landmark(options_.selected_landmark_indices(i));
    }
    *detection = ConvertLandmarksToDetection(subset_landmarks);
  } else {
    *detection = ConvertLandmarksToDetection(landmarks);
  }
  cc->Outputs()
      .Tag(kDetectionTag)
      .Add(detection.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
