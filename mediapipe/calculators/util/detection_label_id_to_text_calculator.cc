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

#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

// Takes a label map (from label IDs to names), and replaces the label IDs
// in Detection protos with label names. Note that the calculator makes a copy
// of the input detections. Consider using it only when the size of input
// detections is small.
//
// Example usage:
// node {
//   calculator: "DetectionLabelIdToTextCalculator"
//   input_stream: "input_detections"
//   output_stream: "output_detections"
//   node_options: {
//     [type.googleapis.com/mediapipe.DetectionLabelIdToTextCalculatorOptions] {
//       label_map_path: "labelmap.txt"
//     }
//   }
// }
class DetectionLabelIdToTextCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  std::unordered_map<int, std::string> label_map_;
};
REGISTER_CALCULATOR(DetectionLabelIdToTextCalculator);

::mediapipe::Status DetectionLabelIdToTextCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::vector<Detection>>();
  cc->Outputs().Index(0).Set<std::vector<Detection>>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status DetectionLabelIdToTextCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options =
      cc->Options<::mediapipe::DetectionLabelIdToTextCalculatorOptions>();

  if (options.has_label_map_path()) {
    std::string string_path;
    ASSIGN_OR_RETURN(string_path,
                     PathToResourceAsFile(options.label_map_path()));
    std::string label_map_string;
    MP_RETURN_IF_ERROR(file::GetContents(string_path, &label_map_string));

    std::istringstream stream(label_map_string);
    std::string line;
    int i = 0;
    while (std::getline(stream, line)) {
      label_map_[i++] = line;
    }
  } else {
    for (int i = 0; i < options.label_size(); ++i) {
      label_map_[i] = options.label(i);
    }
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status DetectionLabelIdToTextCalculator::Process(
    CalculatorContext* cc) {
  std::vector<Detection> output_detections;
  for (const auto& input_detection :
       cc->Inputs().Index(0).Get<std::vector<Detection>>()) {
    output_detections.push_back(input_detection);
    Detection& output_detection = output_detections.back();
    bool has_text_label = false;
    for (const int32 label_id : output_detection.label_id()) {
      if (label_map_.find(label_id) != label_map_.end()) {
        output_detection.add_label(label_map_[label_id]);
        has_text_label = true;
      }
    }
    // Remove label_id field if text labels exist.
    if (has_text_label) {
      output_detection.clear_label_id();
    }
  }
  cc->Outputs().Index(0).AddPacket(
      MakePacket<std::vector<Detection>>(output_detections)
          .At(cc->InputTimestamp()));
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
