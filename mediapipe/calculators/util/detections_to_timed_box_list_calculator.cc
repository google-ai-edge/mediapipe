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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/tracking/box_tracker.h"

namespace mediapipe {

namespace {

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionListTag[] = "DETECTION_LIST";
constexpr char kBoxesTag[] = "BOXES";

}  // namespace

// A calculator that converts Detection proto to TimedBoxList proto for
// tracking.
//
// Please note that only Location Data formats of RELATIVE_BOUNDING_BOX are
// supported.
//
// Example config:
// node {
//   calculator: "DetectionsToTimedBoxListCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "BOXES:boxes"
// }
class DetectionsToTimedBoxListCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kDetectionListTag) ||
              cc->Inputs().HasTag(kDetectionsTag))
        << "None of the input streams are provided.";
    if (cc->Inputs().HasTag(kDetectionListTag)) {
      cc->Inputs().Tag(kDetectionListTag).Set<DetectionList>();
    }
    if (cc->Inputs().HasTag(kDetectionsTag)) {
      cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    }
    cc->Outputs().Tag(kBoxesTag).Set<TimedBoxProtoList>();
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  void ConvertDetectionToTimedBox(const Detection& detection,
                                  TimedBoxProto* box, CalculatorContext* cc);
};
REGISTER_CALCULATOR(DetectionsToTimedBoxListCalculator);

::mediapipe::Status DetectionsToTimedBoxListCalculator::Process(
    CalculatorContext* cc) {
  auto output_timed_box_list = absl::make_unique<TimedBoxProtoList>();

  if (cc->Inputs().HasTag(kDetectionListTag)) {
    const auto& detection_list =
        cc->Inputs().Tag(kDetectionListTag).Get<DetectionList>();
    for (const auto& detection : detection_list.detection()) {
      TimedBoxProto* box = output_timed_box_list->add_box();
      ConvertDetectionToTimedBox(detection, box, cc);
    }
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    const auto& detections =
        cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>();
    for (const auto& detection : detections) {
      TimedBoxProto* box = output_timed_box_list->add_box();
      ConvertDetectionToTimedBox(detection, box, cc);
    }
  }

  cc->Outputs().Tag(kBoxesTag).Add(output_timed_box_list.release(),
                                   cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

void DetectionsToTimedBoxListCalculator::ConvertDetectionToTimedBox(
    const Detection& detection, TimedBoxProto* box, CalculatorContext* cc) {
  const auto& relative_bounding_box =
      detection.location_data().relative_bounding_box();
  box->set_left(relative_bounding_box.xmin());
  box->set_right(relative_bounding_box.xmin() + relative_bounding_box.width());
  box->set_top(relative_bounding_box.ymin());
  box->set_bottom(relative_bounding_box.ymin() +
                  relative_bounding_box.height());
  box->set_id(detection.detection_id());
  box->set_time_msec(cc->InputTimestamp().Microseconds() / 1000);
}

}  // namespace mediapipe
