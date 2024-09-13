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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/proto_ns.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
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
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Local label map built from the calculator options' `label_map_path` or
  // `label` field.
  proto_ns::Map<int64_t, LabelMapItem> local_label_map_;
  bool keep_label_id_;
  const proto_ns::Map<int64_t, LabelMapItem>& GetLabelMap(
      CalculatorContext* cc);
};
REGISTER_CALCULATOR(DetectionLabelIdToTextCalculator);

absl::Status DetectionLabelIdToTextCalculator::GetContract(
    CalculatorContract* cc) {
  cc->Inputs().Index(0).Set<std::vector<Detection>>();
  cc->Outputs().Index(0).Set<std::vector<Detection>>();

  return absl::OkStatus();
}

absl::Status DetectionLabelIdToTextCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  const auto& options = cc->Options<DetectionLabelIdToTextCalculatorOptions>();

  if (options.has_label_map_path()) {
    RET_CHECK(options.label_items().empty() && options.label().empty())
        << "Only can set one of the following fields in the CalculatorOptions: "
           "label_map_path, label, and label_items.";
    std::string string_path;
    MP_ASSIGN_OR_RETURN(string_path,
                        PathToResourceAsFile(options.label_map_path()));
    MP_ASSIGN_OR_RETURN(std::unique_ptr<mediapipe::Resource> label_map,
                        cc->GetResources().Get(string_path));
    MP_ASSIGN_OR_RETURN(
        local_label_map_,
        BuildLabelMapFromFiles(label_map->ToStringView(),
                               /*display_names_file_contents=*/{}));
  } else if (!options.label().empty()) {
    RET_CHECK(options.label_items().empty())
        << "Only can set one of the following fields in the CalculatorOptions: "
           "label_map_path, label, and label_items.";
    for (int i = 0; i < options.label_size(); ++i) {
      LabelMapItem item;
      item.set_name(options.label(i));
      local_label_map_[i] = std::move(item);
    }
  }
  keep_label_id_ = options.keep_label_id();
  return absl::OkStatus();
}

absl::Status DetectionLabelIdToTextCalculator::Process(CalculatorContext* cc) {
  std::vector<Detection> output_detections;
  for (const auto& input_detection :
       cc->Inputs().Index(0).Get<std::vector<Detection>>()) {
    output_detections.push_back(input_detection);
    Detection& output_detection = output_detections.back();
    bool has_text_label = false;
    for (const int32_t label_id : output_detection.label_id()) {
      if (GetLabelMap(cc).contains(label_id)) {
        auto item = GetLabelMap(cc).at(label_id);
        output_detection.add_label(item.name());
        if (item.has_display_name()) {
          output_detection.add_display_name(item.display_name());
        }
        has_text_label = true;
      }
    }
    // Remove label_id field if text labels exist.
    if (has_text_label && !keep_label_id_) {
      output_detection.clear_label_id();
    }
  }
  cc->Outputs().Index(0).AddPacket(
      MakePacket<std::vector<Detection>>(output_detections)
          .At(cc->InputTimestamp()));
  return absl::OkStatus();
}

const proto_ns::Map<int64_t, LabelMapItem>&
DetectionLabelIdToTextCalculator::GetLabelMap(CalculatorContext* cc) {
  return !local_label_map_.empty()
             ? local_label_map_
             : cc->Options<DetectionLabelIdToTextCalculatorOptions>()
                   .label_items();
}

}  // namespace mediapipe
