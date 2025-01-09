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

#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/tool/options_map.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"
#include "mediapipe/tasks/cc/components/processors/embedder_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/embedder_options.pb.h"
#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/image_embedder/proto/image_embedder_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_embedder {

namespace {

constexpr char kEmbeddingsStreamName[] = "embeddings_out";
constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kNormRectTag[] = "NORM_RECT";

constexpr char kGraphTypeName[] =
    "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::components::containers::ConvertToEmbeddingResult;
using ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::tasks::core::PacketMap;
using ::mediapipe::tasks::vision::image_embedder::proto::
    ImageEmbedderGraphOptions;

// Creates a MediaPipe graph config that contains a single node of type
// "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageEmbedderGraphOptions> options_proto,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  auto& task_graph = graph.AddNode(kGraphTypeName);
  task_graph.GetOptions<ImageEmbedderGraphOptions>().Swap(options_proto.get());
  task_graph.Out(kEmbeddingsTag).SetName(kEmbeddingsStreamName) >>
      graph.Out(kEmbeddingsTag);
  task_graph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_graph, {kImageTag, kNormRectTag}, kEmbeddingsTag);
  }
  graph.In(kImageTag) >> task_graph.In(kImageTag);
  graph.In(kNormRectTag) >> task_graph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing ImageEmbedderOptions struct to the internal
// ImageEmbedderGraphOptions proto.
std::unique_ptr<ImageEmbedderGraphOptions> ConvertImageEmbedderOptionsToProto(
    ImageEmbedderOptions* options) {
  auto options_proto = std::make_unique<ImageEmbedderGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  auto embedder_options_proto =
      std::make_unique<components::processors::proto::EmbedderOptions>(
          components::processors::ConvertEmbedderOptionsToProto(
              &(options->embedder_options)));
  options_proto->mutable_embedder_options()->Swap(embedder_options_proto.get());
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageEmbedder>> ImageEmbedder::Create(
    std::unique_ptr<ImageEmbedderOptions> options) {
  auto options_proto = ConvertImageEmbedderOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
          if (!status_or_packets.ok()) {
            Image image;
            result_callback(status_or_packets.status(), image,
                            Timestamp::Unset().Value());
          }
          if (status_or_packets.value()[kImageOutStreamName].IsEmpty()) {
            return;
          }
          Packet embedding_result_packet =
              status_or_packets.value()[kEmbeddingsStreamName];
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          result_callback(ConvertToEmbeddingResult(
                              embedding_result_packet.Get<EmbeddingResult>()),
                          image_packet.Get<Image>(),
                          embedding_result_packet.Timestamp().Value() /
                              kMicroSecondsPerMilliSecond);
        };
  }
  return core::VisionTaskApiFactory::Create<ImageEmbedder,
                                            ImageEmbedderGraphOptions>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<ImageEmbedderResult> ImageEmbedder::Embed(
    Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::StatusOr<ImageEmbedderResult> ImageEmbedder::EmbedForVideo(
    Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return ConvertToEmbeddingResult(
      output_packets[kEmbeddingsStreamName].Get<EmbeddingResult>());
}

absl::Status ImageEmbedder::EmbedAsync(
    Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image));
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kNormRectStreamName,
        MakePacket<NormalizedRect>(std::move(norm_rect))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

absl::StatusOr<double> ImageEmbedder::CosineSimilarity(
    const components::containers::Embedding& u,
    const components::containers::Embedding& v) {
  return components::utils::CosineSimilarity(u, v);
}

}  // namespace image_embedder
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
