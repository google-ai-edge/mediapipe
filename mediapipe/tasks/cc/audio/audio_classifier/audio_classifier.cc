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

#include "mediapipe/tasks/cc/audio/audio_classifier/audio_classifier.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/tasks/cc/audio/audio_classifier/proto/audio_classifier_graph_options.pb.h"
#include "mediapipe/tasks/cc/audio/core/audio_task_api_factory.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace audio_classifier {

namespace {

using ::mediapipe::tasks::components::containers::ConvertToClassificationResult;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;

constexpr char kAudioStreamName[] = "audio_in";
constexpr char kAudioTag[] = "AUDIO";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kClassificationsName[] = "classifications_out";
constexpr char kTimestampedClassificationsTag[] = "TIMESTAMPED_CLASSIFICATIONS";
constexpr char kTimestampedClassificationsName[] =
    "timestamped_classifications_out";
constexpr char kSampleRateName[] = "sample_rate_in";
constexpr char kSampleRateTag[] = "SAMPLE_RATE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// type "AudioClassifierGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<proto::AudioClassifierGraphOptions> options_proto) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  graph.In(kAudioTag).SetName(kAudioStreamName) >> subgraph.In(kAudioTag);
  graph.In(kSampleRateTag).SetName(kSampleRateName) >>
      subgraph.In(kSampleRateTag);
  subgraph.GetOptions<proto::AudioClassifierGraphOptions>().Swap(
      options_proto.get());
  subgraph.Out(kClassificationsTag).SetName(kClassificationsName) >>
      graph.Out(kClassificationsTag);
  subgraph.Out(kTimestampedClassificationsTag)
          .SetName(kTimestampedClassificationsName) >>
      graph.Out(kTimestampedClassificationsTag);
  return graph.GetConfig();
}

// Converts the user-facing AudioClassifierOptions struct to the internal
// AudioClassifierGraphOptions proto.
std::unique_ptr<proto::AudioClassifierGraphOptions>
ConvertAudioClassifierOptionsToProto(AudioClassifierOptions* options) {
  auto options_proto = std::make_unique<proto::AudioClassifierGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode == core::RunningMode::AUDIO_STREAM);
  auto classifier_options_proto =
      std::make_unique<components::processors::proto::ClassifierOptions>(
          components::processors::ConvertClassifierOptionsToProto(
              &(options->classifier_options)));
  options_proto->mutable_classifier_options()->Swap(
      classifier_options_proto.get());
  return options_proto;
}

absl::StatusOr<std::vector<AudioClassifierResult>> ConvertOutputPackets(
    absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
  if (!status_or_packets.ok()) {
    return status_or_packets.status();
  }
  auto classification_results =
      status_or_packets.value()[kTimestampedClassificationsName]
          .Get<std::vector<ClassificationResult>>();
  std::vector<AudioClassifierResult> results;
  results.reserve(classification_results.size());
  for (const auto& classification_result : classification_results) {
    results.emplace_back(ConvertToClassificationResult(classification_result));
  }
  return results;
}

absl::StatusOr<AudioClassifierResult> ConvertAsyncOutputPackets(
    absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
  if (!status_or_packets.ok()) {
    return status_or_packets.status();
  }
  return ConvertToClassificationResult(
      status_or_packets.value()[kClassificationsName]
          .Get<ClassificationResult>());
}
}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<AudioClassifier>> AudioClassifier::Create(
    std::unique_ptr<AudioClassifierOptions> options) {
  auto options_proto = ConvertAudioClassifierOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
          result_callback(ConvertAsyncOutputPackets(status_or_packets));
        };
  }
  return core::AudioTaskApiFactory::Create<AudioClassifier,
                                           proto::AudioClassifierGraphOptions>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<std::vector<AudioClassifierResult>> AudioClassifier::Classify(
    Matrix audio_clip, double audio_sample_rate) {
  return ConvertOutputPackets(ProcessAudioClip(
      {{kAudioStreamName, MakePacket<Matrix>(std::move(audio_clip))},
       {kSampleRateName, MakePacket<double>(audio_sample_rate)}}));
}

absl::Status AudioClassifier::ClassifyAsync(Matrix audio_block,
                                            double audio_sample_rate,
                                            int64_t timestamp_ms) {
  MP_RETURN_IF_ERROR(CheckOrSetSampleRate(kSampleRateName, audio_sample_rate));
  return SendAudioStreamData(
      {{kAudioStreamName,
        MakePacket<Matrix>(std::move(audio_block))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace audio_classifier
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
