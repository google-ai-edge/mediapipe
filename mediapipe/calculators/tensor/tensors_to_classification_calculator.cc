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

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/resource_util.h"
#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {
namespace api2 {
namespace {

void SetClassificationLabel(const LabelMapItem label_map_item,
                            Classification* classification) {
  classification->set_label(label_map_item.name());
  if (label_map_item.has_display_name()) {
    classification->set_display_name(label_map_item.display_name());
  }
}

}  // namespace

// Convert result tensors from classification models into MediaPipe
// classifications.
//
// Input:
//  TENSORS - Vector of Tensors of type kFloat32 containing one
//            tensor, the size of which must be (1, * num_classes).
// Output:
//  CLASSIFICATIONS - Result MediaPipe ClassificationList. The score and index
//                    fields of each classification are set, while the label
//                    field is only set if label_map_path is provided.
//
// Usage example:
// node {
//   calculator: "TensorsToClassificationCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "CLASSIFICATIONS:classifications"
//   options: {
//     [mediapipe.TensorsToClassificationCalculatorOptions.ext] {
//       min_score_threshold: 0.1
//       label_map_path: "labelmap.txt"
//     }
//   }
// }
class TensorsToClassificationCalculator : public Node {
 public:
  static constexpr Input<std::vector<Tensor>> kInTensors{"TENSORS"};
  static constexpr Output<ClassificationList> kOutClassificationList{
      "CLASSIFICATIONS"};
  MEDIAPIPE_NODE_CONTRACT(kInTensors, kOutClassificationList);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  int top_k_ = 0;
  bool sort_by_descending_score_ = false;
  proto_ns::Map<int64, LabelMapItem> local_label_map_;
  bool label_map_loaded_ = false;
  bool is_binary_classification_ = false;
  float min_score_threshold_ = std::numeric_limits<float>::lowest();

  // Set of allowed or ignored class indices.
  struct ClassIndexSet {
    absl::flat_hash_set<int> values;
    bool is_allowlist;
  };
  // Allowed or ignored class indices based on provided options.
  // These are used to filter out the output classification results.
  ClassIndexSet class_index_set_;
  bool IsClassIndexAllowed(int class_index);
  const proto_ns::Map<int64, LabelMapItem>& GetLabelMap(CalculatorContext* cc);
};
MEDIAPIPE_REGISTER_NODE(TensorsToClassificationCalculator);

absl::Status TensorsToClassificationCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<TensorsToClassificationCalculatorOptions>();

  top_k_ = options.top_k();
  sort_by_descending_score_ = options.sort_by_descending_score();
  if (options.has_label_map_path()) {
    std::string string_path;
    ASSIGN_OR_RETURN(string_path,
                     PathToResourceAsFile(options.label_map_path()));
    std::string label_map_string;
    MP_RETURN_IF_ERROR(
        mediapipe::GetResourceContents(string_path, &label_map_string));

    std::istringstream stream(label_map_string);
    std::string line;
    int i = 0;
    while (std::getline(stream, line)) {
      LabelMapItem item;
      item.set_name(line);
      local_label_map_[i++] = item;
    }
    label_map_loaded_ = true;
  } else if (!options.label_items().empty()) {
    label_map_loaded_ = true;
  } else if (options.has_label_map()) {
    for (int i = 0; i < options.label_map().entries_size(); ++i) {
      const auto& entry = options.label_map().entries(i);
      RET_CHECK(!local_label_map_.contains(entry.id()))
          << "Duplicate id found: " << entry.id();
      LabelMapItem item;
      item.set_name(entry.label());
      local_label_map_[entry.id()] = item;
    }
    label_map_loaded_ = true;
  }
  if (options.has_min_score_threshold()) {
    min_score_threshold_ = options.min_score_threshold();
  }
  is_binary_classification_ = options.binary_classification();

  if (is_binary_classification_) {
    RET_CHECK(options.allow_classes().empty() &&
              options.ignore_classes().empty());
  }
  if (!options.allow_classes().empty()) {
    RET_CHECK(options.ignore_classes().empty());
    class_index_set_.is_allowlist = true;
    for (int i = 0; i < options.allow_classes_size(); ++i) {
      class_index_set_.values.insert(options.allow_classes(i));
    }
  } else {
    class_index_set_.is_allowlist = false;
    for (int i = 0; i < options.ignore_classes_size(); ++i) {
      class_index_set_.values.insert(options.ignore_classes(i));
    }
  }

  return absl::OkStatus();
}

absl::Status TensorsToClassificationCalculator::Process(CalculatorContext* cc) {
  const auto& input_tensors = *kInTensors(cc);
  RET_CHECK_EQ(input_tensors.size(), 1);

  int num_classes = input_tensors[0].shape().num_elements();

  if (is_binary_classification_) {
    RET_CHECK_EQ(num_classes, 1);
    // Number of classes for binary classification.
    num_classes = 2;
  }
  if (label_map_loaded_) {
    RET_CHECK_EQ(num_classes, GetLabelMap(cc).size());
  }
  auto view = input_tensors[0].GetCpuReadView();
  auto raw_scores = view.buffer<float>();

  auto classification_list = absl::make_unique<ClassificationList>();
  if (is_binary_classification_) {
    Classification* class_first = classification_list->add_classification();
    Classification* class_second = classification_list->add_classification();
    class_first->set_index(0);
    class_second->set_index(1);
    class_first->set_score(raw_scores[0]);
    class_second->set_score(1. - raw_scores[0]);

    if (label_map_loaded_) {
      SetClassificationLabel(GetLabelMap(cc).at(0), class_first);
      SetClassificationLabel(GetLabelMap(cc).at(1), class_second);
    }
  } else {
    for (int i = 0; i < num_classes; ++i) {
      if (!IsClassIndexAllowed(i)) {
        continue;
      }
      if (raw_scores[i] < min_score_threshold_) {
        continue;
      }
      Classification* classification =
          classification_list->add_classification();
      classification->set_index(i);
      classification->set_score(raw_scores[i]);
      if (label_map_loaded_) {
        SetClassificationLabel(GetLabelMap(cc).at(i), classification);
      }
    }
  }

  auto raw_classification_list = classification_list->mutable_classification();
  if (top_k_ > 0) {
    int desired_size =
        std::min(classification_list->classification_size(), top_k_);
    std::partial_sort(raw_classification_list->begin(),
                      raw_classification_list->begin() + desired_size,
                      raw_classification_list->end(),
                      [](const Classification a, const Classification b) {
                        return a.score() > b.score();
                      });

    if (desired_size >= top_k_) {
      // Resizes the underlying list to have only top_k_ classifications.
      raw_classification_list->DeleteSubrange(
          top_k_, raw_classification_list->size() - top_k_);
    }
  } else if (sort_by_descending_score_) {
    std::sort(raw_classification_list->begin(), raw_classification_list->end(),
              [](const Classification a, const Classification b) {
                return a.score() > b.score();
              });
  }
  kOutClassificationList(cc).Send(std::move(classification_list));
  return absl::OkStatus();
}

absl::Status TensorsToClassificationCalculator::Close(CalculatorContext* cc) {
  return absl::OkStatus();
}

bool TensorsToClassificationCalculator::IsClassIndexAllowed(int class_index) {
  if (class_index_set_.values.empty()) {
    return true;
  }
  if (class_index_set_.is_allowlist) {
    return class_index_set_.values.contains(class_index);
  } else {
    return !class_index_set_.values.contains(class_index);
  }
}

const proto_ns::Map<int64, LabelMapItem>&
TensorsToClassificationCalculator::GetLabelMap(CalculatorContext* cc) {
  return !local_label_map_.empty()
             ? local_label_map_
             : cc->Options<TensorsToClassificationCalculatorOptions>()
                   .label_items();
}

}  // namespace api2
}  // namespace mediapipe
