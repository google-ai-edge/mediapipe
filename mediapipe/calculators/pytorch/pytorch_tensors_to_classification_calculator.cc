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

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "mediapipe/calculators/pytorch/pytorch_tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "torch/torch.h"
#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

namespace {
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kClassificationListTag[] = "CLASSIFICATION_LIST";

using Inputs = torch::Tensor;
}  // namespace

// Convert result PyTorch tensors from classification models into MediaPipe
// classifications.
//
// Input:
//  TENSORS - Vector of PyTorch of type kTfLiteFloat32 containing one
//            tensor, the size of which must be (1, * num_classes).
// Output:
//  CLASSIFICATIONS - Result MediaPipe ClassificationList. The score and index
//                    fields of each classification are set, while the label
//                    field is only set if label_map_path is provided.
//
// Usage example:
// node {
//   calculator: "PyTorchTensorsToClassificationCalculator"
//   input_stream: "TENSORS:tensors"
//   output_stream: "CLASSIFICATIONS:classifications"
//   options: {
//     [mediapipe.PyTorchTensorsToClassificationCalculatorOptions.ext] {
//       num_classes: 1024
//       min_score_threshold: 0.1
//       label_map_path: "labelmap.txt"
//     }
//   }
// }
class PyTorchTensorsToClassificationCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;
  ::mediapipe::Status Process(CalculatorContext* cc) override;
  ::mediapipe::Status Close(CalculatorContext* cc) override;

 private:
  ::mediapipe::PyTorchTensorsToClassificationCalculatorOptions options_;
  std::unordered_map<int, std::string> label_map_;
  bool label_map_loaded_ = false;
};
REGISTER_CALCULATOR(PyTorchTensorsToClassificationCalculator);

::mediapipe::Status PyTorchTensorsToClassificationCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kTensorsTag));
  cc->Inputs().Tag(kTensorsTag).Set<Inputs>();

  RET_CHECK(!cc->Outputs().GetTags().empty());
  if (cc->Outputs().HasTag(kClassificationListTag)) {
    cc->Outputs().Tag(kClassificationListTag).Set<ClassificationList>();
  }

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchTensorsToClassificationCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  options_ = cc->Options<
      ::mediapipe::PyTorchTensorsToClassificationCalculatorOptions>();

  if (options_.has_label_map_path()) {
    std::string string_path;
    ASSIGN_OR_RETURN(string_path,
                     PathToResourceAsFile(options_.label_map_path()));
    std::string label_map_string;
    MP_RETURN_IF_ERROR(file::GetContents(string_path, &label_map_string));

    std::istringstream stream(label_map_string);
    std::string line;
    int i = 0;
    while (std::getline(stream, line)) label_map_[i++] = line;
    label_map_loaded_ = true;
  }

  if (options_.has_top_k()) RET_CHECK_GT(options_.top_k(), 0);

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchTensorsToClassificationCalculator::Process(
    CalculatorContext* cc) {
  const auto& input_tensors = cc->Inputs().Tag(kTensorsTag).Get<Inputs>();
  RET_CHECK_EQ(input_tensors.dim(), 2);

  std::tuple<torch::Tensor, torch::Tensor> result =
      input_tensors.sort(/*dim*/ -1, /*descending*/ true);
  const torch::Tensor scores_tensor = std::get<0>(result)[0];
  const torch::Tensor indices_tensor =
      std::get<1>(result)[0].toType(torch::kInt32);

  auto scores = scores_tensor.accessor<float, 1>();
  auto indices = indices_tensor.accessor<int, 1>();

  const auto indices_count = indices.size(0);
  RET_CHECK_EQ(indices_count, scores.size(0));
  if (label_map_loaded_)
    RET_CHECK_EQ(indices_count, label_map_.size())
        << "need: " << indices_count << ", got: " << label_map_.size();

  // RET_CHECK_GE(indices_count, options_.top_k());
  auto top_k = indices.size(0);
  if (options_.has_top_k()) top_k = options_.top_k();

  auto classification_list = absl::make_unique<ClassificationList>();
  for (int i = 0; i < indices_count; ++i) {
    if (classification_list->classification_size() == top_k) break;
    const float score = scores[i];
    const int index = indices[i];
    if (options_.has_min_score_threshold() &&
        score < options_.min_score_threshold())
      continue;

    Classification* classification = classification_list->add_classification();
    classification->set_score(score);
    classification->set_index(index);
    if (label_map_loaded_) classification->set_label(label_map_[index]);
  }

  cc->Outputs()
      .Tag(kClassificationListTag)
      .Add(classification_list.release(), cc->InputTimestamp());

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PyTorchTensorsToClassificationCalculator::Close(
    CalculatorContext* cc) {
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
