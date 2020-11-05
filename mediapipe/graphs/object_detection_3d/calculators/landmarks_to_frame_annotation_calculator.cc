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

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/object_detection_3d/calculators/annotation_data.pb.h"

namespace mediapipe {

namespace {

constexpr char kInputLandmarksTag[] = "LANDMARKS";
constexpr char kOutputFrameAnnotationTag[] = "FRAME_ANNOTATION";

}  // namespace

// A calculator that converts NormalizedLandmarkList to FrameAnnotation proto.
class LandmarksToFrameAnnotationCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(LandmarksToFrameAnnotationCalculator);

::mediapipe::Status LandmarksToFrameAnnotationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputLandmarksTag)) {
    cc->Inputs().Tag(kInputLandmarksTag).Set<NormalizedLandmarkList>();
  }

  if (cc->Outputs().HasTag(kOutputFrameAnnotationTag)) {
    cc->Outputs().Tag(kOutputFrameAnnotationTag).Set<FrameAnnotation>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFrameAnnotationCalculator::Process(
    CalculatorContext* cc) {
  auto frame_annotation = absl::make_unique<FrameAnnotation>();
  auto* box_annotation = frame_annotation->add_annotations();

  const auto& landmarks =
      cc->Inputs().Tag(kInputLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_GT(landmarks.landmark_size(), 0)
      << "Input landmark vector is empty.";
  for (int i = 0; i < landmarks.landmark_size(); ++i) {
    auto* point2d = box_annotation->add_keypoints()->mutable_point_2d();
    point2d->set_x(landmarks.landmark(i).x());
    point2d->set_y(landmarks.landmark(i).y());
  }
  // Output
  if (cc->Outputs().HasTag(kOutputFrameAnnotationTag)) {
    cc->Outputs()
        .Tag(kOutputFrameAnnotationTag)
        .Add(frame_annotation.release(), cc->InputTimestamp());
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
