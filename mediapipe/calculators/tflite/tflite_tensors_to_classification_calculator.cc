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
#include "mediapipe/calculators/tflite/tflite_tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/resource_util.h"
#include "tensorflow/lite/interpreter.h"
#if defined(__ANDROID__) || (defined(__APPLE__) && !TARGET_OS_OSX)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

// Convert result TFLite tensors from classification models into MediaPipe
// classifications.
//
// Input:
//  TENSORS - Vector of TfLiteTensor of type kTfLiteFloat32 containing one
//            tensor, the size of which must be (1, * num_classes).
// Output:
//  CLASSIFICATIONS - Result MediaPipe ClassificationList. The score and index
//                    fields of each classification are set, while the label
//                    field is only set if label_map_path is provided.
//
// Usage example:
// node {
//   calculator: "TfLiteTensorsToClassificationCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "CLASSIFICATIONS:classifications"
//   options: {
//     [mediapipe.TfLiteTensorsToClassificationCalculatorOptions.ext] {
//       num_classes: 1024
//       min_score_threshold: 0.1
//       label_map_path: "labelmap.txt"
//     }
//   }
// }
class TfLiteTensorsToClassificationCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  int top_k_ = 0;
  double min_score_threshold_ = 0;
  std::unordered_map<int, std::string> label_map_;
  bool label_map_loaded_ = false;
};
REGISTER_CALCULATOR(TfLiteTensorsToClassificationCalculator);

::mediapipe::Status TfLiteTensorsToClassificationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag("TENSORS")) {
    cc->Inputs().Tag("TENSORS").Set<std::vector<TfLiteTensor>>();
  }

  if (cc->Outputs().HasTag("CLASSIFICATIONS")) {
    cc->Outputs().Tag("CLASSIFICATIONS").Set<ClassificationList>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToClassificationCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  auto options = cc->Options<
      ::mediapipe::TfLiteTensorsToClassificationCalculatorOptions>();

  top_k_ = options.top_k();
  min_score_threshold_ = options.min_score_threshold();
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
    label_map_loaded_ = true;
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToClassificationCalculator::Process(
    CalculatorContext* cc) {
  const auto& input_tensors =
      cc->Inputs().Tag("TENSORS").Get<std::vector<TfLiteTensor>>();

  RET_CHECK_EQ(input_tensors.size(), 1);

  const TfLiteTensor* raw_score_tensor = &input_tensors[0];
  RET_CHECK_EQ(raw_score_tensor->dims->size, 2);
  RET_CHECK_EQ(raw_score_tensor->dims->data[0], 1);
  int num_classes = raw_score_tensor->dims->data[1];
  if (label_map_loaded_) {
    RET_CHECK_EQ(num_classes, label_map_.size());
  }
  const float* raw_scores = raw_score_tensor->data.f;

  auto classification_list = absl::make_unique<ClassificationList>();
  for (int i = 0; i < num_classes; ++i) {
    if (raw_scores[i] < min_score_threshold_) {
      continue;
    }
    Classification* classification = classification_list->add_classification();
    classification->set_index(i);
    classification->set_score(raw_scores[i]);
    if (label_map_loaded_) {
      classification->set_label(label_map_[i]);
    }
  }

  // Note that partial_sort will raise error when top_k_ >
  // classification_list->classification_size().
  auto raw_classification_list = classification_list->mutable_classification();
  if (top_k_ > 0 && classification_list->classification_size() >= top_k_) {
    std::partial_sort(raw_classification_list->begin(),
                      raw_classification_list->begin() + top_k_,
                      raw_classification_list->end(),
                      [](const Classification a, const Classification b) {
                        return a.score() > b.score();
                      });

    // Resizes the underlying list to have only top_k_ classifications.
    raw_classification_list->DeleteSubrange(
        top_k_, raw_classification_list->size() - top_k_);
  }
  cc->Outputs()
      .Tag("CLASSIFICATIONS")
      .Add(classification_list.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

::mediapipe::Status TfLiteTensorsToClassificationCalculator::Close(
    CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
