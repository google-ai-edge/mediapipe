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

#include <utility>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/box_util.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace {
constexpr char kInputStreamTag[] = "FRAME_ANNOTATION";
constexpr char kOutputStreamTag[] = "BOXES";
}  // namespace

namespace mediapipe {

// Convert FrameAnnotation 3d bounding box detections to TimedBoxListProto
// 2d bounding boxes.
//
// Input:
//  FRAME_ANNOTATION - 3d bounding box annotation.
// Output:
//  BOXES - 2d bounding box enclosing the projection of 3d box.
//
// Usage example:
// node {
//   calculator: "FrameAnnotationToTimedBoxListCalculator"
//   input_stream: "FRAME_ANNOTATION:frame_annotation"
//   output_stream: "BOXES:boxes"
// }
class FrameAnnotationToTimedBoxListCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(FrameAnnotationToTimedBoxListCalculator);

absl::Status FrameAnnotationToTimedBoxListCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputStreamTag)) {
    cc->Inputs().Tag(kInputStreamTag).Set<FrameAnnotation>();
  }

  if (cc->Outputs().HasTag(kOutputStreamTag)) {
    cc->Outputs().Tag(kOutputStreamTag).Set<TimedBoxProtoList>();
  }
  return absl::OkStatus();
}

absl::Status FrameAnnotationToTimedBoxListCalculator::Open(
    CalculatorContext* cc) {
  return absl::OkStatus();
}

absl::Status FrameAnnotationToTimedBoxListCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kInputStreamTag) &&
      !cc->Inputs().Tag(kInputStreamTag).IsEmpty()) {
    const auto& frame_annotation =
        cc->Inputs().Tag(kInputStreamTag).Get<FrameAnnotation>();
    auto output_objects = absl::make_unique<TimedBoxProtoList>();
    for (const auto& annotation : frame_annotation.annotations()) {
      std::vector<cv::Point2f> key_points;
      for (const auto& keypoint : annotation.keypoints()) {
        key_points.push_back(
            cv::Point2f(keypoint.point_2d().x(), keypoint.point_2d().y()));
      }
      TimedBoxProto* added_box = output_objects->add_box();
      ComputeBoundingRect(key_points, added_box);
      added_box->set_id(annotation.object_id());
      const int64_t time_msec =
          static_cast<int64_t>(std::round(frame_annotation.timestamp() / 1000));
      added_box->set_time_msec(time_msec);
    }

    // Output
    if (cc->Outputs().HasTag(kOutputStreamTag)) {
      cc->Outputs()
          .Tag(kOutputStreamTag)
          .Add(output_objects.release(), cc->InputTimestamp());
    }
  }

  return absl::OkStatus();
}

absl::Status FrameAnnotationToTimedBoxListCalculator::Close(
    CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace mediapipe
