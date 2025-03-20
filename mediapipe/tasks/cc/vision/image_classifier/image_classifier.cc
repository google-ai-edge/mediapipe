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

#include "mediapipe/tasks/cc/vision/image_classifier/image_classifier.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/image_classifier/proto/image_classifier_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_classifier {

namespace {

constexpr char kClassificationsStreamName[] = "classifications_out";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectName[] = "norm_rect_in";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::components::containers::ConvertToClassificationResult;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::tasks::core::PacketMap;

// Creates a MediaPipe graph config that contains a subgraph node of
// type "ImageClassifierGraph". If the task is running in the live stream mode,
// a "FlowLimiterCalculator" will be added to limit the number of frames in
// flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<proto::ImageClassifierGraphOptions> options_proto,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectName);
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<proto::ImageClassifierGraphOptions>().Swap(
      options_proto.get());
  task_subgraph.Out(kClassificationsTag).SetName(kClassificationsStreamName) >>
      graph.Out(kClassificationsTag);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_subgraph, {kImageTag, kNormRectTag}, kClassificationsTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> task_subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing ImageClassifierOptions struct to the internal
// ImageClassifierGraphOptions proto.
std::unique_ptr<proto::ImageClassifierGraphOptions>
ConvertImageClassifierOptionsToProto(ImageClassifierOptions* options) {
  auto options_proto = std::make_unique<proto::ImageClassifierGraphOptions>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  auto classifier_options_proto =
      std::make_unique<components::processors::proto::ClassifierOptions>(
          components::processors::ConvertClassifierOptionsToProto(
              &(options->classifier_options)));
  options_proto->mutable_classifier_options()->Swap(
      classifier_options_proto.get());
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageClassifier>> ImageClassifier::Create(
    std::unique_ptr<ImageClassifierOptions> options) {
  auto options_proto = ConvertImageClassifierOptionsToProto(options.get());
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
          Packet classifications_packet =
              status_or_packets.value()[kClassificationsStreamName];
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          result_callback(
              ConvertToClassificationResult(
                  classifications_packet.Get<ClassificationResult>()),
              image_packet.Get<Image>(),
              classifications_packet.Timestamp().Value() /
                  kMicroSecondsPerMilliSecond);
        };
  }
  return core::VisionTaskApiFactory::Create<ImageClassifier,
                                            proto::ImageClassifierGraphOptions>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<ImageClassifierResult> ImageClassifier::Classify(
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
           {kNormRectName, MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  return ConvertToClassificationResult(
      output_packets[kClassificationsStreamName].Get<ClassificationResult>());
}

absl::StatusOr<ImageClassifierResult> ImageClassifier::ClassifyForVideo(
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
           {kNormRectName,
            MakePacket<NormalizedRect>(std::move(norm_rect))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return ConvertToClassificationResult(
      output_packets[kClassificationsStreamName].Get<ClassificationResult>());
}

absl::Status ImageClassifier::ClassifyAsync(
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
       {kNormRectName,
        MakePacket<NormalizedRect>(std::move(norm_rect))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace image_classifier
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
