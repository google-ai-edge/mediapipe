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
#include <istream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mediapipe/calculators/util/top_k_scores_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/statusor.h"
#include "mediapipe/util/resource_util.h"

#if defined(MEDIAPIPE_MOBILE)
#include "mediapipe/util/android/file/base/file.h"
#include "mediapipe/util/android/file/base/helpers.h"
#else
#include "mediapipe/framework/port/file_helpers.h"
#endif

namespace mediapipe {

// A calculator that takes a vector of scores and returns the indexes, scores,
// labels of the top k elements, classification protos, and summary std::string
// (in csv format).
//
// Usage example:
// node {
//   calculator: "TopKScoresCalculator"
//   input_stream: "SCORES:score_vector"
//   output_stream: "TOP_K_INDEXES:top_k_indexes"
//   output_stream: "TOP_K_SCORES:top_k_scores"
//   output_stream: "TOP_K_LABELS:top_k_labels"
//   output_stream: "TOP_K_CLASSIFICATIONS:top_k_classes"
//   output_stream: "SUMMARY:summary"
//   options: {
//     [mediapipe.TopKScoresCalculatorOptions.ext] {
//       top_k: 5
//       threshold: 0.1
//       label_map_path: "/path/to/label/map"
//     }
//   }
// }
class TopKScoresCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  ::mediapipe::Status LoadLabelmap(std::string label_map_path);

  int top_k_ = -1;
  float threshold_ = 0.0;
  std::unordered_map<int, std::string> label_map_;
  bool label_map_loaded_ = false;
};
REGISTER_CALCULATOR(TopKScoresCalculator);

::mediapipe::Status TopKScoresCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag("SCORES"));
  cc->Inputs().Tag("SCORES").Set<std::vector<float>>();
  if (cc->Outputs().HasTag("TOP_K_INDEXES")) {
    cc->Outputs().Tag("TOP_K_INDEXES").Set<std::vector<int>>();
  }
  if (cc->Outputs().HasTag("TOP_K_SCORES")) {
    cc->Outputs().Tag("TOP_K_SCORES").Set<std::vector<float>>();
  }
  if (cc->Outputs().HasTag("TOP_K_LABELS")) {
    cc->Outputs().Tag("TOP_K_LABELS").Set<std::vector<std::string>>();
  }
  if (cc->Outputs().HasTag("CLASSIFICATIONS")) {
    cc->Outputs().Tag("CLASSIFICATIONS").Set<ClassificationList>();
  }
  if (cc->Outputs().HasTag("SUMMARY")) {
    cc->Outputs().Tag("SUMMARY").Set<std::string>();
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TopKScoresCalculator::Open(CalculatorContext* cc) {
  const auto& options = cc->Options<::mediapipe::TopKScoresCalculatorOptions>();
  RET_CHECK(options.has_top_k() || options.has_threshold())
      << "Must specify at least one of the top_k and threshold fields in "
         "TopKScoresCalculatorOptions.";
  if (options.has_top_k()) {
    RET_CHECK(options.top_k() > 0) << "top_k must be greater than zero.";
    top_k_ = options.top_k();
  }
  if (options.has_threshold()) {
    threshold_ = options.threshold();
  }
  if (options.has_label_map_path()) {
    MP_RETURN_IF_ERROR(LoadLabelmap(options.label_map_path()));
  }
  if (cc->Outputs().HasTag("TOP_K_LABELS")) {
    RET_CHECK(!label_map_.empty());
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TopKScoresCalculator::Process(CalculatorContext* cc) {
  const std::vector<float>& input_vector =
      cc->Inputs().Tag("SCORES").Get<std::vector<float>>();
  std::vector<int> top_k_indexes;

  std::vector<float> top_k_scores;

  std::vector<std::string> top_k_labels;

  if (top_k_ > 0) {
    top_k_indexes.reserve(top_k_);
    top_k_scores.reserve(top_k_);
    top_k_labels.reserve(top_k_);
  }
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      pq;
  for (int i = 0; i < input_vector.size(); ++i) {
    if (input_vector[i] < threshold_) {
      continue;
    }
    if (top_k_ > 0) {
      if (pq.size() < top_k_) {
        pq.push(std::pair<float, int>(input_vector[i], i));
      } else if (pq.top().first < input_vector[i]) {
        pq.pop();
        pq.push(std::pair<float, int>(input_vector[i], i));
      }
    } else {
      pq.push(std::pair<float, int>(input_vector[i], i));
    }
  }

  while (!pq.empty()) {
    top_k_indexes.push_back(pq.top().second);
    top_k_scores.push_back(pq.top().first);
    pq.pop();
  }
  reverse(top_k_indexes.begin(), top_k_indexes.end());
  reverse(top_k_scores.begin(), top_k_scores.end());

  if (label_map_loaded_) {
    for (int index : top_k_indexes) {
      top_k_labels.push_back(label_map_[index]);
    }
  }
  if (cc->Outputs().HasTag("TOP_K_INDEXES")) {
    cc->Outputs()
        .Tag("TOP_K_INDEXES")
        .AddPacket(MakePacket<std::vector<int>>(top_k_indexes)
                       .At(cc->InputTimestamp()));
  }
  if (cc->Outputs().HasTag("TOP_K_SCORES")) {
    cc->Outputs()
        .Tag("TOP_K_SCORES")
        .AddPacket(MakePacket<std::vector<float>>(top_k_scores)
                       .At(cc->InputTimestamp()));
  }
  if (cc->Outputs().HasTag("TOP_K_LABELS")) {
    cc->Outputs()
        .Tag("TOP_K_LABELS")
        .AddPacket(MakePacket<std::vector<std::string>>(top_k_labels)
                       .At(cc->InputTimestamp()));
  }

  if (cc->Outputs().HasTag("SUMMARY")) {
    std::vector<std::string> results;
    for (int index = 0; index < top_k_indexes.size(); ++index) {
      if (label_map_loaded_) {
        results.push_back(
            absl::StrCat(top_k_labels[index], ":", top_k_scores[index]));
      } else {
        results.push_back(
            absl::StrCat(top_k_indexes[index], ":", top_k_scores[index]));
      }
    }
    cc->Outputs().Tag("SUMMARY").AddPacket(
        MakePacket<std::string>(absl::StrJoin(results, ","))
            .At(cc->InputTimestamp()));
  }

  if (cc->Outputs().HasTag("TOP_K_CLASSIFICATION")) {
    auto classification_list = absl::make_unique<ClassificationList>();
    for (int index = 0; index < top_k_indexes.size(); ++index) {
      Classification* classification =
          classification_list->add_classification();
      classification->set_index(top_k_indexes[index]);
      classification->set_score(top_k_scores[index]);
      if (label_map_loaded_) {
        classification->set_label(top_k_labels[index]);
      }
    }
  }
  return ::mediapipe::OkStatus();
}

::mediapipe::Status TopKScoresCalculator::LoadLabelmap(
    std::string label_map_path) {
  std::string string_path;
  ASSIGN_OR_RETURN(string_path, PathToResourceAsFile(label_map_path));
  std::string label_map_string;
  MP_RETURN_IF_ERROR(file::GetContents(string_path, &label_map_string));

  std::istringstream stream(label_map_string);
  std::string line;
  int i = 0;
  while (std::getline(stream, line)) {
    label_map_[i++] = line;
  }
  label_map_loaded_ = true;
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
