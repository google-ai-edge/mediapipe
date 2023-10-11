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

#include "mediapipe/tasks/cc/audio/audio_embedder/audio_embedder.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/tasks/cc/audio/audio_embedder/proto/audio_embedder_graph_options.pb.h"
#include "mediapipe/tasks/cc/audio/core/audio_task_api_factory.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe::tasks::audio::audio_embedder {
namespace {
using ::mediapipe::tasks::components::containers::ConvertToEmbeddingResult;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
constexpr char kAudioStreamName[] = "audio_in";
constexpr char kAudioTag[] = "AUDIO";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kTimestampedEmbeddingsTag[] = "TIMESTAMPED_EMBEDDINGS";
constexpr char kEmbeddingsName[] = "embeddings_out";
constexpr char kTimestampedEmbeddingsName[] = "timestamped_embeddings_out";
constexpr char kSampleRateName[] = "sample_rate_in";
constexpr char kSampleRateTag[] = "SAMPLE_RATE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// type "AudioEmbedderGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<proto::AudioEmbedderGraphOptions> options_proto) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  graph.In(kAudioTag).SetName(kAudioStreamName) >> subgraph.In(kAudioTag);
  graph.In(kSampleRateTag).SetName(kSampleRateName) >>
      subgraph.In(kSampleRateTag);
  subgraph.GetOptions<proto::AudioEmbedderGraphOptions>().Swap(
      options_proto.get());
  subgraph.Out(kEmbeddingsTag).SetName(kEmbeddingsName) >>
      graph.Out(kEmbeddingsTag);
  subgraph.Out(kTimestampedEmbeddingsTag).SetName(kTimestampedEmbeddingsName) >>
      graph.Out(kTimestampedEmbeddingsTag);
  return graph.GetConfig();
}

// Converts the user-facing AudioEmbedderOptions struct to the internal
// AudioEmbedderGraphOptions proto.
std::unique_ptr<proto::AudioEmbedderGraphOptions>
ConvertAudioEmbedderOptionsToProto(AudioEmbedderOptions* options) {
  auto options_proto = std::make_unique<proto::AudioEmbedderGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode == core::RunningMode::AUDIO_STREAM);
  auto embedder_options_proto =
      std::make_unique<components::processors::proto::EmbedderOptions>(
          components::processors::ConvertEmbedderOptionsToProto(
              &(options->embedder_options)));
  options_proto->mutable_embedder_options()->Swap(embedder_options_proto.get());
  return options_proto;
}

absl::StatusOr<std::vector<AudioEmbedderResult>> ConvertOutputPackets(
    absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
  if (!status_or_packets.ok()) {
    return status_or_packets.status();
  }
  auto embedding_results = status_or_packets.value()[kTimestampedEmbeddingsName]
                               .Get<std::vector<EmbeddingResult>>();
  std::vector<AudioEmbedderResult> results;
  results.reserve(embedding_results.size());
  for (const auto& embedding_result : embedding_results) {
    results.emplace_back(ConvertToEmbeddingResult(embedding_result));
  }
  return results;
}

absl::StatusOr<AudioEmbedderResult> ConvertAsyncOutputPackets(
    absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
  if (!status_or_packets.ok()) {
    return status_or_packets.status();
  }
  return ConvertToEmbeddingResult(
      status_or_packets.value()[kEmbeddingsName].Get<EmbeddingResult>());
}
}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<AudioEmbedder>> AudioEmbedder::Create(
    std::unique_ptr<AudioEmbedderOptions> options) {
  auto options_proto = ConvertAudioEmbedderOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
          result_callback(ConvertAsyncOutputPackets(status_or_packets));
        };
  }
  return core::AudioTaskApiFactory::Create<AudioEmbedder,
                                           proto::AudioEmbedderGraphOptions>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<std::vector<AudioEmbedderResult>> AudioEmbedder::Embed(
    Matrix audio_clip, double audio_sample_rate) {
  return ConvertOutputPackets(ProcessAudioClip(
      {{kAudioStreamName, MakePacket<Matrix>(std::move(audio_clip))},
       {kSampleRateName, MakePacket<double>(audio_sample_rate)}}));
}

absl::Status AudioEmbedder::EmbedAsync(Matrix audio_block,
                                       double audio_sample_rate,
                                       int64_t timestamp_ms) {
  MP_RETURN_IF_ERROR(CheckOrSetSampleRate(kSampleRateName, audio_sample_rate));
  return SendAudioStreamData(
      {{kAudioStreamName,
        MakePacket<Matrix>(std::move(audio_block))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace mediapipe::tasks::audio::audio_embedder
