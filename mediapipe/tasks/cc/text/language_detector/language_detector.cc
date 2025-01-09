/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"
#include "mediapipe/tasks/cc/text/text_classifier/proto/text_classifier_graph_options.pb.h"

namespace mediapipe::tasks::text::language_detector {

namespace {

using ::mediapipe::tasks::components::containers::Category;
using ::mediapipe::tasks::components::containers::ClassificationResult;
using ::mediapipe::tasks::components::containers::Classifications;
using ::mediapipe::tasks::components::containers::ConvertToClassificationResult;
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::tasks::text::text_classifier::proto::
    TextClassifierGraphOptions;

constexpr char kTextStreamName[] = "text_in";
constexpr char kTextTag[] = "TEXT";
constexpr char kClassificationsStreamName[] = "classifications_out";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.text.text_classifier.TextClassifierGraph";

// Creates a MediaPipe graph config that only contains a single subgraph node of
// type "TextClassifierGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<TextClassifierGraphOptions> options) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  subgraph.GetOptions<TextClassifierGraphOptions>().Swap(options.get());
  graph.In(kTextTag).SetName(kTextStreamName) >> subgraph.In(kTextTag);
  subgraph.Out(kClassificationsTag).SetName(kClassificationsStreamName) >>
      graph.Out(kClassificationsTag);
  return graph.GetConfig();
}

// Converts the user-facing LanguageDetectorOptions struct to the internal
// TextClassifierGraphOptions proto.
std::unique_ptr<TextClassifierGraphOptions>
ConvertLanguageDetectorOptionsToProto(LanguageDetectorOptions* options) {
  auto options_proto = std::make_unique<TextClassifierGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  auto classifier_options_proto =
      std::make_unique<tasks::components::processors::proto::ClassifierOptions>(
          components::processors::ConvertClassifierOptionsToProto(
              &(options->classifier_options)));
  options_proto->mutable_classifier_options()->Swap(
      classifier_options_proto.get());
  return options_proto;
}

absl::StatusOr<LanguageDetectorResult>
ExtractLanguageDetectorResultFromClassificationResult(
    const ClassificationResult& classification_result) {
  if (classification_result.classifications.size() != 1) {
    return absl::InvalidArgumentError(
        "The LanguageDetector TextClassifierGraph should have exactly one "
        "classification head.");
  }
  const Classifications& languages_and_scores =
      classification_result.classifications[0];
  LanguageDetectorResult language_detector_result;
  for (const Category& category : languages_and_scores.categories) {
    if (!category.category_name.has_value()) {
      return absl::InvalidArgumentError(
          "LanguageDetector ClassificationResult has a missing language code.");
    }
    language_detector_result.push_back(
        {.language_code = *category.category_name,
         .probability = category.score});
  }
  return language_detector_result;
}

}  // namespace

absl::StatusOr<std::unique_ptr<LanguageDetector>> LanguageDetector::Create(
    std::unique_ptr<LanguageDetectorOptions> options) {
  auto options_proto = ConvertLanguageDetectorOptionsToProto(options.get());
  return core::TaskApiFactory::Create<LanguageDetector,
                                      TextClassifierGraphOptions>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver));
}

absl::StatusOr<LanguageDetectorResult> LanguageDetector::Detect(
    absl::string_view text) {
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextStreamName, MakePacket<std::string>(std::string(text))}}));
  ClassificationResult classification_result =
      ConvertToClassificationResult(output_packets[kClassificationsStreamName]
                                        .Get<ClassificationResultProto>());
  return ExtractLanguageDetectorResultFromClassificationResult(
      classification_result);
}

}  // namespace mediapipe::tasks::text::language_detector
