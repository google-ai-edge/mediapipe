/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/collection.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.pb.h"

namespace mediapipe {
namespace api2 {
namespace {

constexpr char kPredictionTag[] = "PREDICTION";

Classification GetMaxScoringClassification(
    const ClassificationList& classifications) {
  Classification max_classification;
  max_classification.set_score(0);
  for (const auto& input : classifications.classification()) {
    if (max_classification.score() < input.score()) {
      max_classification = input;
    }
  }
  return max_classification;
}

float GetScoreThreshold(
    const std::string& input_label,
    const absl::btree_map<std::string, float>& classwise_thresholds,
    const std::string& background_label, const float default_threshold) {
  float threshold = default_threshold;
  auto it = classwise_thresholds.find(input_label);
  if (it != classwise_thresholds.end()) {
    threshold = it->second;
  }
  return threshold;
}

std::unique_ptr<ClassificationList> GetWinningPrediction(
    const ClassificationList& classification_list,
    const absl::btree_map<std::string, float>& classwise_thresholds,
    const std::string& background_label, const float default_threshold) {
  auto prediction_list = std::make_unique<ClassificationList>();
  if (classification_list.classification().empty()) {
    return prediction_list;
  }
  Classification& prediction = *prediction_list->add_classification();
  auto argmax_prediction = GetMaxScoringClassification(classification_list);
  float argmax_prediction_thresh =
      GetScoreThreshold(argmax_prediction.label(), classwise_thresholds,
                        background_label, default_threshold);
  if (argmax_prediction.score() >= argmax_prediction_thresh) {
    prediction.set_label(argmax_prediction.label());
    prediction.set_score(argmax_prediction.score());
  } else {
    for (const auto& input : classification_list.classification()) {
      if (input.label() == background_label) {
        prediction.set_label(input.label());
        prediction.set_score(input.score());
        break;
      }
    }
  }
  return prediction_list;
}

}  // namespace

// This calculator accepts multiple ClassificationList input streams. Each
// ClassificationList should contain classifications with labels and
// corresponding softmax scores. The calculator computes the best prediction for
// each ClassificationList input stream via argmax and thresholding. Thresholds
// for all classes can be specified in the
// `CombinedPredictionCalculatorOptions`, along with a default global
// threshold.
// Please note that for this calculator to work as designed, the class names
// other than the background class in the ClassificationList objects must be
// different, but the background class name has to be the same. This background
// label name can be set via `background_label` in
// `CombinedPredictionCalculatorOptions`.
// The ClassificationList in the PREDICTION output stream contains the label of
// the winning class and corresponding softmax score. If none of the
// ClassificationList objects has a non-background winning class, the output
// contains the background class and score of the background class in the first
// ClassificationList. If multiple ClassificationList objects have a
// non-background winning class, the output contains the winning prediction from
// the ClassificationList with the highest priority. Priority is in decreasing
// order of input streams to the graph node using this calculator.
// Input:
//   At least one stream with ClassificationList.
// Output:
//  PREDICTION - A ClassificationList with the winning label as the only item.
//
// Usage example:
// node {
//   calculator: "CombinedPredictionCalculator"
//   input_stream: "classification_list_0"
//   input_stream: "classification_list_1"
//   output_stream: "PREDICTION:prediction"
//   options {
//     [mediapipe.CombinedPredictionCalculatorOptions.ext] {
//       class {
//         label: "A"
//         score_threshold: 0.7
//       }
//       default_global_threshold: 0.1
//       background_label: "B"
//     }
//   }
// }

class CombinedPredictionCalculator : public Node {
 public:
  static constexpr Input<ClassificationList>::Multiple kClassificationListIn{
      ""};
  static constexpr Output<ClassificationList> kPredictionOut{"PREDICTION"};
  MEDIAPIPE_NODE_CONTRACT(kClassificationListIn, kPredictionOut);

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<CombinedPredictionCalculatorOptions>();
    for (const auto& input : options_.class_()) {
      classwise_thresholds_[input.label()] = input.score_threshold();
    }
    classwise_thresholds_[options_.background_label()] = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    // After loop, if have winning prediction return. Otherwise empty packet.
    std::unique_ptr<ClassificationList> first_winning_prediction = nullptr;
    auto collection = kClassificationListIn(cc);
    for (const auto& input : collection) {
      if (input.IsEmpty() || input.Get().classification_size() == 0) {
        continue;
      }
      auto prediction = GetWinningPrediction(
          input.Get(), classwise_thresholds_, options_.background_label(),
          options_.default_global_threshold());
      if (prediction->classification(0).label() !=
          options_.background_label()) {
        kPredictionOut(cc).Send(std::move(prediction));
        return absl::OkStatus();
      }
      if (first_winning_prediction == nullptr) {
        first_winning_prediction = std::move(prediction);
      }
    }
    if (first_winning_prediction != nullptr) {
      kPredictionOut(cc).Send(std::move(first_winning_prediction));
    }
    return absl::OkStatus();
  }

 private:
  CombinedPredictionCalculatorOptions options_;
  absl::btree_map<std::string, float> classwise_thresholds_;
};

MEDIAPIPE_REGISTER_NODE(CombinedPredictionCalculator);

}  // namespace api2
}  // namespace mediapipe
