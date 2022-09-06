/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/tasks/cc/audio/audio_classifier/proto/audio_classifier_options.pb.h"
#include "mediapipe/tasks/cc/audio/core/audio_task_api_factory.h"
#include "mediapipe/tasks/cc/components/classifier_options.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace {

constexpr char kAudioStreamName[] = "audio_in";
constexpr char kAudioTag[] = "AUDIO";
constexpr char kClassificationResultStreamName[] = "classification_result_out";
constexpr char kClassificationResultTag[] = "CLASSIFICATION_RESULT";
constexpr char kSampleRateName[] = "sample_rate_in";
constexpr char kSampleRateTag[] = "SAMPLE_RATE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.audio.AudioClassifierGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using AudioClassifierOptionsProto =
    audio_classifier::proto::AudioClassifierOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.audio.AudioClassifierGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<AudioClassifierOptionsProto> options_proto) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  graph.In(kAudioTag).SetName(kAudioStreamName) >> subgraph.In(kAudioTag);
  if (!options_proto->base_options().use_stream_mode()) {
    graph.In(kSampleRateTag).SetName(kSampleRateName) >>
        subgraph.In(kSampleRateTag);
  }
  subgraph.GetOptions<AudioClassifierOptionsProto>().Swap(options_proto.get());
  subgraph.Out(kClassificationResultTag)
          .SetName(kClassificationResultStreamName) >>
      graph.Out(kClassificationResultTag);
  return graph.GetConfig();
}

// Converts the user-facing AudioClassifierOptions struct to the internal
// AudioClassifierOptions proto.
std::unique_ptr<AudioClassifierOptionsProto>
ConvertAudioClassifierOptionsToProto(AudioClassifierOptions* options) {
  auto options_proto = std::make_unique<AudioClassifierOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode == core::RunningMode::AUDIO_STREAM);
  auto classifier_options_proto = std::make_unique<tasks::ClassifierOptions>(
      components::ConvertClassifierOptionsToProto(
          &(options->classifier_options)));
  options_proto->mutable_classifier_options()->Swap(
      classifier_options_proto.get());
  if (options->sample_rate > 0) {
    options_proto->set_default_input_audio_sample_rate(options->sample_rate);
  }
  return options_proto;
}

absl::StatusOr<ClassificationResult> ConvertOutputPackets(
    absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
  if (!status_or_packets.ok()) {
    return status_or_packets.status();
  }
  return status_or_packets.value()[kClassificationResultStreamName]
      .Get<ClassificationResult>();
}
}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<AudioClassifier>> AudioClassifier::Create(
    std::unique_ptr<AudioClassifierOptions> options) {
  if (options->running_mode == core::RunningMode::AUDIO_STREAM &&
      options->sample_rate < 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "The audio classifier is in audio stream mode, the sample rate must be "
        "specified in the AudioClassifierOptions.",
        MediaPipeTasksStatus::kInvalidTaskGraphConfigError);
  }
  auto options_proto = ConvertAudioClassifierOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
          result_callback(ConvertOutputPackets(status_or_packets));
        };
  }
  return core::AudioTaskApiFactory::Create<AudioClassifier,
                                           AudioClassifierOptionsProto>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<ClassificationResult> AudioClassifier::Classify(
    Matrix audio_clip, double audio_sample_rate) {
  return ConvertOutputPackets(ProcessAudioClip(
      {{kAudioStreamName, MakePacket<Matrix>(std::move(audio_clip))},
       {kSampleRateName, MakePacket<double>(audio_sample_rate)}}));
}

absl::Status AudioClassifier::ClassifyAsync(Matrix audio_block,
                                            int64 timestamp_ms) {
  return SendAudioStreamData(
      {{kAudioStreamName,
        MakePacket<Matrix>(std::move(audio_block))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
