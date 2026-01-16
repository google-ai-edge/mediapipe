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

#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"
#include "mediapipe/tasks/cc/text/text_classifier/proto/text_classifier_graph_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace text_classifier {

namespace {

using ::mediapipe::tasks::components::containers::ConvertToClassificationResult;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;

constexpr char kTextStreamName[] = "text_in";
constexpr char kTextTag[] = "TEXT";
constexpr char kClassificationsStreamName[] = "classifications_out";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.text.text_classifier.TextClassifierGraph";

// Creates a MediaPipe graph config that only contains a single subgraph node of
// type "TextClassifierGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<proto::TextClassifierGraphOptions> options) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  subgraph.GetOptions<proto::TextClassifierGraphOptions>().Swap(options.get());
  graph.In(kTextTag).SetName(kTextStreamName) >> subgraph.In(kTextTag);
  subgraph.Out(kClassificationsTag).SetName(kClassificationsStreamName) >>
      graph.Out(kClassificationsTag);
  return graph.GetConfig();
}

// Converts the user-facing TextClassifierOptions struct to the internal
// TextClassifierGraphOptions proto.
std::unique_ptr<proto::TextClassifierGraphOptions>
ConvertTextClassifierOptionsToProto(TextClassifierOptions* options) {
  auto options_proto = std::make_unique<proto::TextClassifierGraphOptions>();
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

}  // namespace

absl::StatusOr<std::unique_ptr<TextClassifier>> TextClassifier::Create(
    std::unique_ptr<TextClassifierOptions> options) {
  auto options_proto = ConvertTextClassifierOptionsToProto(options.get());
  return core::TaskApiFactory::Create<TextClassifier,
                                      proto::TextClassifierGraphOptions>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver));
}

absl::StatusOr<TextClassifierResult> TextClassifier::Classify(
    absl::string_view text) {
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      runner_->Process(
          {{kTextStreamName, MakePacket<std::string>(std::string(text))}}));
  return ConvertToClassificationResult(
      output_packets[kClassificationsStreamName].Get<ClassificationResult>());
}

}  // namespace text_classifier
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
