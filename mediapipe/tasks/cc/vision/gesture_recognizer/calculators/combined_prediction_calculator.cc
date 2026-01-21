/* Copyright 2025 The MediaPipe Authors.

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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/calculators/combined_prediction_calculator.pb.h"

namespace mediapipe {
namespace tasks {
namespace {

using ::mediapipe::api3::Calculator;
using ::mediapipe::api3::CalculatorContext;

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

class CombinedPredictionNodeImpl
    : public Calculator<CombinedPredictionNode, CombinedPredictionNodeImpl> {
 public:
  absl::Status Open(CalculatorContext<CombinedPredictionNode>& cc) override {
    options_ = cc.options.Get();
    for (const auto& input : options_.class_()) {
      classwise_thresholds_[input.label()] = input.score_threshold();
    }
    classwise_thresholds_[options_.background_label()] = 0;
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext<CombinedPredictionNode>& cc) override {
    // After loop, if have winning prediction return. Otherwise empty packet.
    std::unique_ptr<ClassificationList> first_winning_prediction = nullptr;
    const int count = cc.classification_list_in.Count();
    for (int i = 0; i < count; ++i) {
      const auto& input = cc.classification_list_in.At(i);
      if (!input) {
        continue;
      }
      const ClassificationList& classification_list = input.GetOrDie();
      if (classification_list.classification_size() == 0) {
        continue;
      }
      std::unique_ptr<ClassificationList> prediction = GetWinningPrediction(
          classification_list, classwise_thresholds_,
          options_.background_label(), options_.default_global_threshold());
      if (prediction->classification(0).label() !=
          options_.background_label()) {
        cc.prediction_out.Send(std::move(prediction));
        return absl::OkStatus();
      }
      if (first_winning_prediction == nullptr) {
        first_winning_prediction = std::move(prediction);
      }
    }

    if (first_winning_prediction != nullptr) {
      cc.prediction_out.Send(std::move(first_winning_prediction));
    }

    return absl::OkStatus();
  }

 private:
  CombinedPredictionCalculatorOptions options_;
  absl::btree_map<std::string, float> classwise_thresholds_;
};

}  // namespace tasks
}  // namespace mediapipe
