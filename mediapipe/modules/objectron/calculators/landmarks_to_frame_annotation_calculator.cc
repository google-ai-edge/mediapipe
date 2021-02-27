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
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"

namespace mediapipe {

namespace {

constexpr char kInputLandmarksTag[] = "LANDMARKS";
constexpr char kInputMultiLandmarksTag[] = "MULTI_LANDMARKS";
constexpr char kOutputFrameAnnotationTag[] = "FRAME_ANNOTATION";

}  // namespace

// A calculator that converts NormalizedLandmarkList to FrameAnnotation proto.
class LandmarksToFrameAnnotationCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  void AddLandmarksToFrameAnnotation(const NormalizedLandmarkList& landmarks,
                                     FrameAnnotation* frame_annotation);
};
REGISTER_CALCULATOR(LandmarksToFrameAnnotationCalculator);

absl::Status LandmarksToFrameAnnotationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputLandmarksTag)) {
    cc->Inputs().Tag(kInputLandmarksTag).Set<NormalizedLandmarkList>();
  }
  if (cc->Inputs().HasTag(kInputMultiLandmarksTag)) {
    cc->Inputs()
        .Tag(kInputMultiLandmarksTag)
        .Set<std::vector<NormalizedLandmarkList>>();
  }
  if (cc->Outputs().HasTag(kOutputFrameAnnotationTag)) {
    cc->Outputs().Tag(kOutputFrameAnnotationTag).Set<FrameAnnotation>();
  }
  return absl::OkStatus();
}

absl::Status LandmarksToFrameAnnotationCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status LandmarksToFrameAnnotationCalculator::Process(
    CalculatorContext* cc) {
  auto frame_annotation = absl::make_unique<FrameAnnotation>();

  // Handle the case when input has only one NormalizedLandmarkList.
  if (cc->Inputs().HasTag(kInputLandmarksTag) &&
      !cc->Inputs().Tag(kInputLandmarksTag).IsEmpty()) {
    const auto& landmarks =
        cc->Inputs().Tag(kInputMultiLandmarksTag).Get<NormalizedLandmarkList>();
    AddLandmarksToFrameAnnotation(landmarks, frame_annotation.get());
  }

  // Handle the case when input has muliple NormalizedLandmarkList.
  if (cc->Inputs().HasTag(kInputMultiLandmarksTag) &&
      !cc->Inputs().Tag(kInputMultiLandmarksTag).IsEmpty()) {
    const auto& landmarks_list =
        cc->Inputs()
            .Tag(kInputMultiLandmarksTag)
            .Get<std::vector<NormalizedLandmarkList>>();
    for (const auto& landmarks : landmarks_list) {
      AddLandmarksToFrameAnnotation(landmarks, frame_annotation.get());
    }
  }

  // Output
  if (cc->Outputs().HasTag(kOutputFrameAnnotationTag)) {
    cc->Outputs()
        .Tag(kOutputFrameAnnotationTag)
        .Add(frame_annotation.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

void LandmarksToFrameAnnotationCalculator::AddLandmarksToFrameAnnotation(
    const NormalizedLandmarkList& landmarks,
    FrameAnnotation* frame_annotation) {
  auto* new_annotation = frame_annotation->add_annotations();
  for (const auto& landmark : landmarks.landmark()) {
    auto* point2d = new_annotation->add_keypoints()->mutable_point_2d();
    point2d->set_x(landmark.x());
    point2d->set_y(landmark.y());
  }
}

}  // namespace mediapipe
