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
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::components::containers::proto::ClassificationResult;

// Aggregates ClassificationLists into either a ClassificationResult object
// representing the classification results aggregated by classifier head, or
// into an std::vector<ClassificationResult> representing the classification
// results aggregated first by timestamp then by classifier head.
//
// Inputs:
//   CLASSIFICATIONS - ClassificationList @Multiple
//     ClassificationList per classification head.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of the timestamps that this calculator should aggregate.
//     This stream is optional: if provided then the TIMESTAMPED_CLASSIFICATIONS
//     output is used for results. Otherwise as no timestamp aggregation is
//     required the CLASSIFICATIONS output is used for results.
//
// Outputs:
//   CLASSIFICATIONS - ClassificationResult @Optional
//     The classification results aggregated by head. Must be connected if the
//     TIMESTAMPS input is not connected, as it signals that timestamp
//     aggregation is not required.
//   TIMESTAMPED_CLASSIFICATIONS - std::vector<ClassificationResult> @Optional
//     The classification result aggregated by timestamp, then by head. Must be
//     connected if the TIMESTAMPS input is connected, as it signals that
//     timestamp aggregation is required.
//
// Example without timestamp aggregation:
// node {
//   calculator: "ClassificationAggregationCalculator"
//   input_stream: "CLASSIFICATIONS:0:stream_a"
//   input_stream: "CLASSIFICATIONS:1:stream_b"
//   input_stream: "CLASSIFICATIONS:2:stream_c"
//   output_stream: "CLASSIFICATIONS:classifications"
//   options {
//    [mediapipe.ClassificationAggregationCalculatorOptions.ext] {
//      head_names: "head_name_a"
//      head_names: "head_name_b"
//      head_names: "head_name_c"
//    }
//  }
// }
//
// Example with timestamp aggregation:
// node {
//   calculator: "ClassificationAggregationCalculator"
//   input_stream: "CLASSIFICATIONS:0:stream_a"
//   input_stream: "CLASSIFICATIONS:1:stream_b"
//   input_stream: "CLASSIFICATIONS:2:stream_c"
//   input_stream: "TIMESTAMPS:timestamps"
//   output_stream: "TIMESTAMPED_CLASSIFICATIONS:timestamped_classifications"
//   options {
//    [mediapipe.ClassificationAggregationCalculatorOptions.ext] {
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
  static constexpr Output<ClassificationResult>::Optional kClassificationsOut{
      "CLASSIFICATIONS"};
  static constexpr Output<std::vector<ClassificationResult>>::Optional
      kTimestampedClassificationsOut{"TIMESTAMPED_CLASSIFICATIONS"};
  static constexpr Output<ClassificationResult>::Optional
      kClassificationResultOut{"CLASSIFICATION_RESULT"};
  MEDIAPIPE_NODE_CONTRACT(kClassificationListIn, kTimestampsIn,
                          kClassificationsOut, kTimestampedClassificationsOut,
                          kClassificationResultOut);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc);
  absl::Status Process(CalculatorContext* cc);

 private:
  std::vector<std::string> head_names_;
  bool time_aggregation_enabled_;
  std::unordered_map<int64_t, std::vector<ClassificationList>>
      cached_classifications_;

  ClassificationResult ConvertToClassificationResult(CalculatorContext* cc);
  std::vector<ClassificationResult> ConvertToTimestampedClassificationResults(
      CalculatorContext* cc);
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
  if (kTimestampsIn(cc).IsConnected()) {
    RET_CHECK(kTimestampedClassificationsOut(cc).IsConnected());
  } else {
    RET_CHECK(kClassificationsOut(cc).IsConnected());
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
  ClassificationResult classification_result;
  if (time_aggregation_enabled_) {
    if (kTimestampsIn(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    kTimestampedClassificationsOut(cc).Send(
        ConvertToTimestampedClassificationResults(cc));
  } else {
    kClassificationsOut(cc).Send(ConvertToClassificationResult(cc));
  }
  kClassificationResultOut(cc).Send(classification_result);
  RET_CHECK(cached_classifications_.empty());
  return absl::OkStatus();
}

ClassificationResult
ClassificationAggregationCalculator::ConvertToClassificationResult(
    CalculatorContext* cc) {
  ClassificationResult result;
  auto& classification_lists =
      cached_classifications_[cc->InputTimestamp().Value()];
  for (int i = 0; i < classification_lists.size(); ++i) {
    auto classifications = result.add_classifications();
    classifications->set_head_index(i);
    if (!head_names_.empty()) {
      classifications->set_head_name(head_names_[i]);
    }
    *classifications->mutable_classification_list() =
        std::move(classification_lists[i]);
  }
  result.set_timestamp_ms(cc->InputTimestamp().Value() / 1000);
  cached_classifications_.erase(cc->InputTimestamp().Value());
  return result;
}

std::vector<ClassificationResult>
ClassificationAggregationCalculator::ConvertToTimestampedClassificationResults(
    CalculatorContext* cc) {
  auto timestamps = kTimestampsIn(cc).Get();
  std::vector<ClassificationResult> results;
  results.reserve(timestamps.size());
  for (const auto& timestamp : timestamps) {
    ClassificationResult result;
    result.set_timestamp_ms((timestamp.Value() - timestamps[0].Value()) / 1000);
    auto& classification_lists = cached_classifications_[timestamp.Value()];
    for (int i = 0; i < classification_lists.size(); ++i) {
      auto classifications = result.add_classifications();
      classifications->set_head_index(i);
      if (!head_names_.empty()) {
        classifications->set_head_name(head_names_[i]);
      }
      *classifications->mutable_classification_list() =
          std::move(classification_lists[i]);
    }
    cached_classifications_.erase(timestamp.Value());
    results.push_back(std::move(result));
  }
  return results;
}

MEDIAPIPE_REGISTER_NODE(ClassificationAggregationCalculator);

}  // namespace api2
}  // namespace mediapipe
