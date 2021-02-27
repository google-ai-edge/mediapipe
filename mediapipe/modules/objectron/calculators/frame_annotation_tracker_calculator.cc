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

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/annotation_data.pb.h"
#include "mediapipe/modules/objectron/calculators/frame_annotation_tracker.h"
#include "mediapipe/modules/objectron/calculators/frame_annotation_tracker_calculator.pb.h"
#include "mediapipe/util/tracking/box_tracker.pb.h"

namespace {
constexpr char kInputFrameAnnotationTag[] = "FRAME_ANNOTATION";
constexpr char kInputTrackedBoxesTag[] = "TRACKED_BOXES";
constexpr char kOutputTrackedFrameAnnotationTag[] = "TRACKED_FRAME_ANNOTATION";
constexpr char kOutputCancelObjectIdTag[] = "CANCEL_OBJECT_ID";
}  // namespace

namespace mediapipe {

// Tracks frame annotations seeded/updated by FRAME_ANNOTATION input_stream.
// When using this calculator, make sure FRAME_ANNOTATION and TRACKED_BOXES
// are in different sync set.
//
// Input:
//  FRAME_ANNOTATION - frame annotation.
//  TRACKED_BOXES - 2d box tracking result
// Output:
//  TRACKED_FRAME_ANNOTATION - annotation inferred from 2d tracking result.
//  CANCEL_OBJECT_ID - object id that needs to be cancelled from the tracker.
//
// Usage example:
// node {
//   calculator: "FrameAnnotationTrackerCalculator"
//   input_stream: "FRAME_ANNOTATION:frame_annotation"
//   input_stream: "TRACKED_BOXES:tracked_boxes"
//   output_stream: "TRACKED_FRAME_ANNOTATION:tracked_frame_annotation"
//   output_stream: "CANCEL_OBJECT_ID:cancel_object_id"
// }
class FrameAnnotationTrackerCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  std::unique_ptr<FrameAnnotationTracker> frame_annotation_tracker_;
};
REGISTER_CALCULATOR(FrameAnnotationTrackerCalculator);

absl::Status FrameAnnotationTrackerCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kInputFrameAnnotationTag)) {
    cc->Inputs().Tag(kInputFrameAnnotationTag).Set<FrameAnnotation>();
  }
  if (cc->Inputs().HasTag(kInputTrackedBoxesTag)) {
    cc->Inputs().Tag(kInputTrackedBoxesTag).Set<TimedBoxProtoList>();
  }
  if (cc->Outputs().HasTag(kOutputTrackedFrameAnnotationTag)) {
    cc->Outputs().Tag(kOutputTrackedFrameAnnotationTag).Set<FrameAnnotation>();
  }
  if (cc->Outputs().HasTag(kOutputCancelObjectIdTag)) {
    cc->Outputs().Tag(kOutputCancelObjectIdTag).Set<int>();
  }
  return absl::OkStatus();
}

absl::Status FrameAnnotationTrackerCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<FrameAnnotationTrackerCalculatorOptions>();
  frame_annotation_tracker_ = absl::make_unique<FrameAnnotationTracker>(
      options.iou_threshold(), options.img_width(), options.img_height());
  return absl::OkStatus();
}

absl::Status FrameAnnotationTrackerCalculator::Process(CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kInputFrameAnnotationTag) &&
      !cc->Inputs().Tag(kInputFrameAnnotationTag).IsEmpty()) {
    frame_annotation_tracker_->AddDetectionResult(
        cc->Inputs().Tag(kInputFrameAnnotationTag).Get<FrameAnnotation>());
  }
  if (cc->Inputs().HasTag(kInputTrackedBoxesTag) &&
      !cc->Inputs().Tag(kInputTrackedBoxesTag).IsEmpty() &&
      cc->Outputs().HasTag(kOutputTrackedFrameAnnotationTag)) {
    absl::flat_hash_set<int> cancel_object_ids;
    auto output_frame_annotation = absl::make_unique<FrameAnnotation>();
    *output_frame_annotation =
        frame_annotation_tracker_->ConsolidateTrackingResult(
            cc->Inputs().Tag(kInputTrackedBoxesTag).Get<TimedBoxProtoList>(),
            &cancel_object_ids);
    output_frame_annotation->set_timestamp(cc->InputTimestamp().Microseconds());

    cc->Outputs()
        .Tag(kOutputTrackedFrameAnnotationTag)
        .Add(output_frame_annotation.release(), cc->InputTimestamp());

    if (cc->Outputs().HasTag(kOutputCancelObjectIdTag)) {
      auto packet_timestamp = cc->InputTimestamp();
      for (const auto& id : cancel_object_ids) {
        // The timestamp is incremented (by 1 us) because currently the box
        // tracker calculator only accepts one cancel object ID for any given
        // timestamp.
        cc->Outputs()
            .Tag(kOutputCancelObjectIdTag)
            .AddPacket(mediapipe::MakePacket<int>(id).At(packet_timestamp++));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status FrameAnnotationTrackerCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

}  // namespace mediapipe
