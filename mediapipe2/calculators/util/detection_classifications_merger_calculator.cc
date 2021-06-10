// Copyright 2021 The MediaPipe Authors.
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

#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"

namespace mediapipe {
namespace api2 {

namespace {}  // namespace

// Replaces the classification labels and scores from the input `Detection` with
// the ones provided into the input `ClassificationList`. Namely:
// * `label_id[i]` becomes `classification[i].index`
// * `score[i]` becomes `classification[i].score`
// * `label[i]` becomes `classification[i].label` (if present)
//
// In case the input `ClassificationList` contains no results (i.e.
// `classification` is empty, which may happen if the classifier uses a score
// threshold and no confident enough result were returned), the input
// `Detection` is returned unchanged.
//
// This is specifically designed for two-stage detection cascades where the
// detections returned by a standalone detector (typically a class-agnostic
// localizer) are fed e.g. into a `TfLiteTaskImageClassifierCalculator` through
// the optional "RECT" or "NORM_RECT" input, e.g:
//
// node {
//   calculator: "DetectionsToRectsCalculator"
//   # Output of an upstream object detector.
//   input_stream: "DETECTION:detection"
//   output_stream: "NORM_RECT:norm_rect"
// }
// node {
//   calculator: "TfLiteTaskImageClassifierCalculator"
//   input_stream: "IMAGE:image"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "CLASSIFICATION_RESULT:classification_result"
// }
// node {
//   calculator: "TfLiteTaskClassificationResultToClassificationsCalculator"
//   input_stream: "CLASSIFICATION_RESULT:classification_result"
//   output_stream: "CLASSIFICATION_LIST:classification_list"
// }
// node {
//   calculator: "DetectionClassificationsMergerCalculator"
//   input_stream: "INPUT_DETECTION:detection"
//   input_stream: "CLASSIFICATION_LIST:classification_list"
//   # Final output.
//   output_stream: "OUTPUT_DETECTION:classified_detection"
// }
//
// Inputs:
// INPUT_DETECTION: `Detection` proto.
// CLASSIFICATION_LIST: `ClassificationList` proto.
//
// Output:
// OUTPUT_DETECTION: modified `Detection` proto.
class DetectionClassificationsMergerCalculator : public Node {
 public:
  static constexpr Input<Detection> kInputDetection{"INPUT_DETECTION"};
  static constexpr Input<ClassificationList> kClassificationList{
      "CLASSIFICATION_LIST"};
  static constexpr Output<Detection> kOutputDetection{"OUTPUT_DETECTION"};

  MEDIAPIPE_NODE_CONTRACT(kInputDetection, kClassificationList,
                          kOutputDetection);

  absl::Status Process(CalculatorContext* cc) override;
};
MEDIAPIPE_REGISTER_NODE(DetectionClassificationsMergerCalculator);

absl::Status DetectionClassificationsMergerCalculator::Process(
    CalculatorContext* cc) {
  if (kInputDetection(cc).IsEmpty() && kClassificationList(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  RET_CHECK(!kInputDetection(cc).IsEmpty());
  RET_CHECK(!kClassificationList(cc).IsEmpty());

  Detection detection = *kInputDetection(cc);
  const ClassificationList& classification_list = *kClassificationList(cc);

  // Update input detection only if classification did return results.
  if (classification_list.classification_size() != 0) {
    detection.clear_label_id();
    detection.clear_score();
    detection.clear_label();
    detection.clear_display_name();
    for (const auto& classification : classification_list.classification()) {
      if (!classification.has_index()) {
        return absl::InvalidArgumentError(
            "Missing required 'index' field in Classification proto.");
      }
      detection.add_label_id(classification.index());
      if (!classification.has_score()) {
        return absl::InvalidArgumentError(
            "Missing required 'score' field in Classification proto.");
      }
      detection.add_score(classification.score());
      if (classification.has_label()) {
        detection.add_label(classification.label());
      }
      if (classification.has_display_name()) {
        detection.add_display_name(classification.display_name());
      }
    }
    // Post-conversion sanity checks.
    if (detection.label_size() != 0 &&
        detection.label_size() != detection.label_id_size()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Each input Classification is expected to either always or never "
          "provide a 'label' field. Found $0 'label' fields for $1 "
          "'Classification' objects.",
          /*$0=*/detection.label_size(), /*$1=*/detection.label_id_size()));
    }
    if (detection.display_name_size() != 0 &&
        detection.display_name_size() != detection.label_id_size()) {
      return absl::InvalidArgumentError(absl::Substitute(
          "Each input Classification is expected to either always or never "
          "provide a 'display_name' field. Found $0 'display_name' fields for "
          "$1 'Classification' objects.",
          /*$0=*/detection.display_name_size(),
          /*$1=*/detection.label_id_size()));
    }
  }
  kOutputDetection(cc).Send(detection);
  return absl::OkStatus();
}

}  // namespace api2
}  // namespace mediapipe
