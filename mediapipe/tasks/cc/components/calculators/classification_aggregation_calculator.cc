// Copyright 2022 The MediaPipe Authors.
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
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/components/calculators/classification_aggregation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::ClassificationAggregationCalculatorOptions;
using ::mediapipe::tasks::ClassificationResult;
using ::mediapipe::tasks::Classifications;

// Aggregates ClassificationLists into a single ClassificationResult that has
// 3 dimensions: (classification head, classification timestamp, classification
// category).
//
// Inputs:
//   CLASSIFICATIONS - ClassificationList
//     ClassificationList per classification head.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of the timestamps that a single ClassificationResult
//     should aggragate. This stream is optional, and the timestamp information
//     will only be populated to the ClassificationResult proto when this stream
//     is connected.
//
// Outputs:
//   CLASSIFICATION_RESULT - ClassificationResult
//     The aggregated classification result.
//
// Example:
// node {
//   calculator: "ClassificationAggregationCalculator"
//   input_stream: "CLASSIFICATIONS:0:stream_a"
//   input_stream: "CLASSIFICATIONS:1:stream_b"
//   input_stream: "CLASSIFICATIONS:2:stream_c"
//   input_stream: "TIMESTAMPS:timestamps"
//   output_stream: "CLASSIFICATION_RESULT:classification_result"
//   options {
//    [mediapipe.tasks.ClassificationAggregationCalculatorOptions.ext] {
//      head_names: "head_name_a"
//      head_names: "head_name_b"
//      head_names: "head_name_c"
//    }
//  }
// }
class ClassificationAggregationCalculator : public Node {
 public:
  static constexpr Input<ClassificationList>::Multiple kClassificationListIn{
      "CLASSIFICATIONS"};
  static constexpr Input<std::vector<Timestamp>>::Optional kTimestampsIn{
      "TIMESTAMPS"};
  static constexpr Output<ClassificationResult> kOut{"CLASSIFICATION_RESULT"};
  MEDIAPIPE_NODE_CONTRACT(kClassificationListIn, kTimestampsIn, kOut);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);

 private:
  std::vector<std::string> head_names_;
  bool time_aggregation_enabled_;
  std::unordered_map<int64, std::vector<ClassificationList>>
      cached_classifications_;

  ClassificationResult ConvertToClassificationResult(CalculatorContext* cc);
};

absl::Status ClassificationAggregationCalculator::UpdateContract(
    CalculatorContract* cc) {
  RET_CHECK_GE(kClassificationListIn(cc).Count(), 1);
  const auto& options =
      cc->Options<ClassificationAggregationCalculatorOptions>();
  if (!options.head_names().empty()) {
    RET_CHECK_EQ(kClassificationListIn(cc).Count(), options.head_names().size())
        << "The size of classifications input streams should match the "
           "size of head names specified in the calculator options";
  }
  return absl::OkStatus();
}

absl::Status ClassificationAggregationCalculator::Open(CalculatorContext* cc) {
  time_aggregation_enabled_ = kTimestampsIn(cc).IsConnected();
  const auto& options =
      cc->Options<ClassificationAggregationCalculatorOptions>();
  if (!options.head_names().empty()) {
    head_names_.assign(options.head_names().begin(),
                       options.head_names().end());
  }
  return absl::OkStatus();
}

absl::Status ClassificationAggregationCalculator::Process(
    CalculatorContext* cc) {
  std::vector<ClassificationList> classification_lists;
  classification_lists.resize(kClassificationListIn(cc).Count());
  std::transform(
      kClassificationListIn(cc).begin(), kClassificationListIn(cc).end(),
      classification_lists.begin(),
      [](const auto& elem) -> ClassificationList { return elem.Get(); });
  cached_classifications_[cc->InputTimestamp().Value()] =
      std::move(classification_lists);
  if (time_aggregation_enabled_ && kTimestampsIn(cc).IsEmpty()) {
    return absl::OkStatus();
  }
  kOut(cc).Send(ConvertToClassificationResult(cc));
  RET_CHECK(cached_classifications_.empty());
  return absl::OkStatus();
}

ClassificationResult
ClassificationAggregationCalculator::ConvertToClassificationResult(
    CalculatorContext* cc) {
  ClassificationResult result;
  Timestamp first_timestamp(0);
  std::vector<Timestamp> timestamps;
  if (time_aggregation_enabled_) {
    timestamps = kTimestampsIn(cc).Get();
    first_timestamp = timestamps[0];
  } else {
    timestamps = {cc->InputTimestamp()};
  }
  for (Timestamp timestamp : timestamps) {
    int count = cached_classifications_[timestamp.Value()].size();
    for (int i = 0; i < count; ++i) {
      Classifications* c;
      if (result.classifications_size() <= i) {
        c = result.add_classifications();
        if (!head_names_.empty()) {
          c->set_head_index(i);
          c->set_head_name(head_names_[i]);
        }
      } else {
        c = result.mutable_classifications(i);
      }
      auto* entry = c->add_entries();
      for (const auto& elem :
           cached_classifications_[timestamp.Value()][i].classification()) {
        auto* category = entry->add_categories();
        if (elem.has_index()) {
          category->set_index(elem.index());
        }
        if (elem.has_score()) {
          category->set_score(elem.score());
        }
        if (elem.has_label()) {
          category->set_category_name(elem.label());
        }
        if (elem.has_display_name()) {
          category->set_display_name(elem.display_name());
        }
      }
      entry->set_timestamp_ms((timestamp.Value() - first_timestamp.Value()) /
                              1000);
    }
    cached_classifications_.erase(timestamp.Value());
  }
  return result;
}

MEDIAPIPE_REGISTER_NODE(ClassificationAggregationCalculator);

}  // namespace api2
}  // namespace mediapipe
