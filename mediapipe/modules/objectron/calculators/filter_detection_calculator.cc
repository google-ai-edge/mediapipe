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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/node_hash_set.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/map_util.h"
#include "mediapipe/framework/port/re2.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/objectron/calculators/filter_detection_calculator.pb.h"

namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kLabelsTag[] = "LABELS";
constexpr char kLabelsCsvTag[] = "LABELS_CSV";

using mediapipe::RE2;
using Detections = std::vector<Detection>;
using Strings = std::vector<std::string>;

struct FirstGreaterComparator {
  bool operator()(const std::pair<float, int>& a,
                  const std::pair<float, int>& b) const {
    return a.first > b.first;
  }
};

absl::Status SortLabelsByDecreasingScore(const Detection& detection,
                                         Detection* sorted_detection) {
  RET_CHECK(sorted_detection);
  RET_CHECK_EQ(detection.score_size(), detection.label_size());
  if (!detection.label_id().empty()) {
    RET_CHECK_EQ(detection.score_size(), detection.label_id_size());
  }
  // Copies input to keep all fields unchanged, and to reserve space for
  // repeated fields. Repeated fields (score, label, and label_id) will be
  // overwritten.
  *sorted_detection = detection;

  std::vector<std::pair<float, int>> scores_and_indices(detection.score_size());
  for (int i = 0; i < detection.score_size(); ++i) {
    scores_and_indices[i].first = detection.score(i);
    scores_and_indices[i].second = i;
  }

  std::sort(scores_and_indices.begin(), scores_and_indices.end(),
            FirstGreaterComparator());

  for (int i = 0; i < detection.score_size(); ++i) {
    const int index = scores_and_indices[i].second;
    sorted_detection->set_score(i, detection.score(index));
    sorted_detection->set_label(i, detection.label(index));
  }

  if (!detection.label_id().empty()) {
    for (int i = 0; i < detection.score_size(); ++i) {
      const int index = scores_and_indices[i].second;
      sorted_detection->set_label_id(i, detection.label_id(index));
    }
  }
  return absl::OkStatus();
}

}  // namespace

// Filters the entries in a Detection to only those with valid scores
// for the specified allowed labels. Allowed labels are provided as a
// std::vector<std::string> in an optional input side packet. Allowed labels can
// contain simple strings or regular expressions. The valid score range
// can be set in the options.The allowed labels can be provided as
// std::vector<std::string> (LABELS) or CSV string (LABELS_CSV) containing class
// names of allowed labels. Note: Providing an empty vector in the input side
// packet Packet causes this calculator to act as a sink if
// empty_allowed_labels_means_allow_everything is set to false (default value).
// To allow all labels, use the calculator with no input side packet stream, or
// set empty_allowed_labels_means_allow_everything to true.
//
// Example config:
// node {
//   calculator: "FilterDetectionCalculator"
//   input_stream: "DETECTIONS:detections"
//   output_stream: "DETECTIONS:filtered_detections"
//   input_side_packet: "LABELS:allowed_labels"
//   options: {
//     [mediapipe.FilterDetectionCalculatorOptions.ext]: {
//       min_score: 0.5
//     }
//   }
// }

class FilterDetectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  bool IsValidLabel(const std::string& label);
  bool IsValidScore(float score);
  // Stores numeric limits for filtering on the score.
  FilterDetectionCalculatorOptions options_;
  // We use the next two fields to possibly filter to a limited set of
  // classes.  The hash_set will be empty in two cases: 1) if no input
  // side packet stream is provided (not filtering on labels), or 2)
  // if the input side packet contains an empty vector (no labels are
  // allowed). We use limit_labels_ to distinguish between the two cases.
  bool limit_labels_ = true;
  absl::node_hash_set<std::string> allowed_labels_;
};
REGISTER_CALCULATOR(FilterDetectionCalculator);

absl::Status FilterDetectionCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK(!cc->Inputs().GetTags().empty());
  RET_CHECK(!cc->Outputs().GetTags().empty());

  if (cc->Inputs().HasTag(kDetectionTag)) {
    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
    cc->Outputs().Tag(kDetectionTag).Set<Detection>();
  }
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    cc->Inputs().Tag(kDetectionsTag).Set<Detections>();
    cc->Outputs().Tag(kDetectionsTag).Set<Detections>();
  }
  if (cc->InputSidePackets().HasTag(kLabelsTag)) {
    cc->InputSidePackets().Tag(kLabelsTag).Set<Strings>();
  }
  if (cc->InputSidePackets().HasTag(kLabelsCsvTag)) {
    cc->InputSidePackets().Tag(kLabelsCsvTag).Set<std::string>();
  }
  return absl::OkStatus();
}

absl::Status FilterDetectionCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<FilterDetectionCalculatorOptions>();
  limit_labels_ = cc->InputSidePackets().HasTag(kLabelsTag) ||
                  cc->InputSidePackets().HasTag(kLabelsCsvTag);
  if (limit_labels_) {
    Strings allowlist_labels;
    if (cc->InputSidePackets().HasTag(kLabelsCsvTag)) {
      allowlist_labels = absl::StrSplit(
          cc->InputSidePackets().Tag(kLabelsCsvTag).Get<std::string>(), ',',
          absl::SkipWhitespace());
      for (auto& e : allowlist_labels) {
        absl::StripAsciiWhitespace(&e);
      }
    } else {
      allowlist_labels = cc->InputSidePackets().Tag(kLabelsTag).Get<Strings>();
    }
    allowed_labels_.insert(allowlist_labels.begin(), allowlist_labels.end());
  }
  if (limit_labels_ && allowed_labels_.empty()) {
    if (options_.fail_on_empty_labels()) {
      cc->GetCounter("VideosWithEmptyLabelsAllowlist")->Increment();
      return tool::StatusFail(
          "FilterDetectionCalculator received empty allowlist with "
          "fail_on_empty_labels = true.");
    }
    if (options_.empty_allowed_labels_means_allow_everything()) {
      // Continue as if side_input was not provided, i.e. pass all labels.
      limit_labels_ = false;
    }
  }
  return absl::OkStatus();
}

absl::Status FilterDetectionCalculator::Process(CalculatorContext* cc) {
  if (limit_labels_ && allowed_labels_.empty()) {
    return absl::OkStatus();
  }
  Detections detections;
  if (cc->Inputs().HasTag(kDetectionsTag)) {
    detections = cc->Inputs().Tag(kDetectionsTag).Get<Detections>();
  } else if (cc->Inputs().HasTag(kDetectionTag)) {
    detections.emplace_back(cc->Inputs().Tag(kDetectionTag).Get<Detection>());
  }
  std::unique_ptr<Detections> outputs(new Detections);
  for (const auto& input : detections) {
    Detection output;
    for (int i = 0; i < input.label_size(); ++i) {
      const std::string& label = input.label(i);
      const float score = input.score(i);
      if (IsValidLabel(label) && IsValidScore(score)) {
        output.add_label(label);
        output.add_score(score);
      }
    }
    if (output.label_size() > 0) {
      if (input.has_location_data()) {
        *output.mutable_location_data() = input.location_data();
      }
      Detection output_sorted;
      if (!SortLabelsByDecreasingScore(output, &output_sorted).ok()) {
        // Uses the orginal output if fails to sort.
        cc->GetCounter("FailedToSortLabelsInDetection")->Increment();
        output_sorted = output;
      }
      outputs->emplace_back(output_sorted);
    }
  }

  if (cc->Outputs().HasTag(kDetectionsTag)) {
    cc->Outputs()
        .Tag(kDetectionsTag)
        .Add(outputs.release(), cc->InputTimestamp());
  } else if (!outputs->empty()) {
    cc->Outputs()
        .Tag(kDetectionTag)
        .Add(new Detection((*outputs)[0]), cc->InputTimestamp());
  }
  return absl::OkStatus();
}

bool FilterDetectionCalculator::IsValidLabel(const std::string& label) {
  bool match = !limit_labels_ || allowed_labels_.contains(label);
  if (!match) {
    // If no exact match is found, check for regular expression
    // comparions in the allowed_labels.
    for (const auto& label_regexp : allowed_labels_) {
      match = match || RE2::FullMatch(label, RE2(label_regexp));
    }
  }
  return match;
}

bool FilterDetectionCalculator::IsValidScore(float score) {
  if (options_.has_min_score() && score < options_.min_score()) {
    LOG(ERROR) << "Filter out detection with low score " << score;
    return false;
  }
  if (options_.has_max_score() && score > options_.max_score()) {
    LOG(ERROR) << "Filter out detection with high score " << score;
    return false;
  }
  return true;
}

}  // namespace mediapipe
