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
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

namespace {

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionListTag[] = "DETECTION_LIST";

// Each detection processed by DetectionUniqueIDCalculator will be assigned an
// unique id that starts from 1. If a detection already has an ID other than 0,
// the ID will be overwritten.
static int64 detection_id = 0;

inline int GetNextDetectionId() { return ++detection_id; }

}  // namespace

// Assign a unique id to detections.
// Note that the calculator will consume the input vector of Detection or
// DetectionList. So the input stream can not be connected to other calculators.
//
// Example config:
// node {
//   calculator: "DetectionUniqueIdCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "DETECTIONS:output_detections"
// }
class DetectionUniqueIdCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kDetectionListTag) ||
              cc->Inputs().HasTag(kDetectionsTag))
        << "None of the input streams are provided.";

    if (cc->Inputs().HasTag(kDetectionListTag)) {
      RET_CHECK(cc->Outputs().HasTag(kDetectionListTag));
      cc->Inputs().Tag(kDetectionListTag).Set<DetectionList>();
      cc->Outputs().Tag(kDetectionListTag).Set<DetectionList>();
    }
    if (cc->Inputs().HasTag(kDetectionsTag)) {
      RET_CHECK(cc->Outputs().HasTag(kDetectionsTag));
      cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
      cc->Outputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
    }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(::mediapipe::TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(DetectionUniqueIdCalculator);

::mediapipe::Status DetectionUniqueIdCalculator::Process(
    CalculatorContext* cc) {
  if (cc->Inputs().HasTag(kDetectionListTag) &&
      !cc->Inputs().Tag(kDetectionListTag).IsEmpty()) {
    auto result =
        cc->Inputs().Tag(kDetectionListTag).Value().Consume<DetectionList>();
    if (result.ok()) {
      auto detection_list = std::move(result).ValueOrDie();
      for (Detection& detection : *detection_list->mutable_detection()) {
        detection.set_detection_id(GetNextDetectionId());
      }
      cc->Outputs()
          .Tag(kDetectionListTag)
          .Add(detection_list.release(), cc->InputTimestamp());
    }
  }

  if (cc->Inputs().HasTag(kDetectionsTag) &&
      !cc->Inputs().Tag(kDetectionsTag).IsEmpty()) {
    auto result = cc->Inputs()
                      .Tag(kDetectionsTag)
                      .Value()
                      .Consume<std::vector<Detection>>();
    if (result.ok()) {
      auto detections = std::move(result).ValueOrDie();
      for (Detection& detection : *detections) {
        detection.set_detection_id(GetNextDetectionId());
      }
      cc->Outputs()
          .Tag(kDetectionsTag)
          .Add(detections.release(), cc->InputTimestamp());
    }
  }
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
