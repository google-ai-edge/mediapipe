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

#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"

#include <utility>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace hand_landmarker {

namespace {

using HandLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    hand_landmarker::proto::HandLandmarkerGraphOptions;

using ::mediapipe::NormalizedRect;

constexpr char kHandLandmarkerGraphTypeName[] =
    "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessStreamName[] = "handedness";
constexpr char kHandLandmarksTag[] = "LANDMARKS";
constexpr char kHandLandmarksStreamName[] = "landmarks";
constexpr char kHandWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandWorldLandmarksStreamName[] = "world_landmarks";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.hand_ladnamrker.HandLandmarkerGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<HandLandmarkerGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHandLandmarkerGraphTypeName);
  subgraph.GetOptions<HandLandmarkerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  subgraph.Out(kHandednessTag).SetName(kHandednessStreamName) >>
      graph.Out(kHandednessTag);
  subgraph.Out(kHandLandmarksTag).SetName(kHandLandmarksStreamName) >>
      graph.Out(kHandLandmarksTag);
  subgraph.Out(kHandWorldLandmarksTag).SetName(kHandWorldLandmarksStreamName) >>
      graph.Out(kHandWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, subgraph, {kImageTag, kNormRectTag}, kHandLandmarksTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing HandLandmarkerOptions struct to the internal
// HandLandmarkerGraphOptions proto.
std::unique_ptr<HandLandmarkerGraphOptionsProto>
ConvertHandLandmarkerGraphOptionsProto(HandLandmarkerOptions* options) {
  auto options_proto = std::make_unique<HandLandmarkerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure hand detector options.
  auto* hand_detector_graph_options =
      options_proto->mutable_hand_detector_graph_options();
  hand_detector_graph_options->set_num_hands(options->num_hands);
  hand_detector_graph_options->set_min_detection_confidence(
      options->min_hand_detection_confidence);

  // Configure hand landmark detector options.
  options_proto->set_min_tracking_confidence(options->min_tracking_confidence);
  auto* hand_landmarks_detector_graph_options =
      options_proto->mutable_hand_landmarks_detector_graph_options();
  hand_landmarks_detector_graph_options->set_min_detection_confidence(
      options->min_hand_presence_confidence);

  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HandLandmarker>> HandLandmarker::Create(
    std::unique_ptr<HandLandmarkerOptions> options) {
  auto options_proto = ConvertHandLandmarkerGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback = [=](absl::StatusOr<tasks::core::PacketMap>
                               status_or_packets) {
      if (!status_or_packets.ok()) {
        Image image;
        result_callback(status_or_packets.status(), image,
                        Timestamp::Unset().Value());
        return;
      }
      if (status_or_packets.value()[kImageOutStreamName].IsEmpty()) {
        return;
      }
      Packet image_packet = status_or_packets.value()[kImageOutStreamName];
      if (status_or_packets.value()[kHandLandmarksStreamName].IsEmpty()) {
        Packet empty_packet =
            status_or_packets.value()[kHandLandmarksStreamName];
        result_callback(
            {HandLandmarkerResult()}, image_packet.Get<Image>(),
            empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        return;
      }
      Packet handedness_packet =
          status_or_packets.value()[kHandednessStreamName];
      Packet hand_landmarks_packet =
          status_or_packets.value()[kHandLandmarksStreamName];
      Packet hand_world_landmarks_packet =
          status_or_packets.value()[kHandWorldLandmarksStreamName];
      result_callback(
          ConvertToHandLandmarkerResult(
              /* handedness= */ handedness_packet
                  .Get<std::vector<ClassificationList>>(),
              /* hand_landmarks= */
              hand_landmarks_packet.Get<std::vector<NormalizedLandmarkList>>(),
              /* hand_world_landmarks= */
              hand_world_landmarks_packet.Get<std::vector<LandmarkList>>()),
          image_packet.Get<Image>(),
          hand_landmarks_packet.Timestamp().Value() /
              kMicroSecondsPerMilliSecond);
    };
  }
  return core::VisionTaskApiFactory::Create<HandLandmarker,
                                            HandLandmarkerGraphOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<HandLandmarkerResult> HandLandmarker::Detect(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image,
                                              /*roi_allowed=*/false));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  if (output_packets[kHandLandmarksStreamName].IsEmpty()) {
    return {HandLandmarkerResult()};
  }
  return ConvertToHandLandmarkerResult(/* handedness= */
                                       output_packets[kHandednessStreamName]
                                           .Get<std::vector<
                                               mediapipe::
                                                   ClassificationList>>(),
                                       /* hand_landmarks= */
                                       output_packets[kHandLandmarksStreamName]
                                           .Get<std::vector<
                                               mediapipe::
                                                   NormalizedLandmarkList>>(),
                                       /* hand_world_landmarks */
                                       output_packets
                                           [kHandWorldLandmarksStreamName]
                                               .Get<std::vector<
                                                   mediapipe::LandmarkList>>());
}

absl::StatusOr<HandLandmarkerResult> HandLandmarker::DetectForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image,
                                              /*roi_allowed=*/false));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  if (output_packets[kHandLandmarksStreamName].IsEmpty()) {
    return {HandLandmarkerResult()};
  }
  return ConvertToHandLandmarkerResult(/* handedness= */
                                       output_packets[kHandednessStreamName]
                                           .Get<std::vector<
                                               mediapipe::
                                                   ClassificationList>>(),
                                       /* hand_landmarks= */
                                       output_packets[kHandLandmarksStreamName]
                                           .Get<std::vector<
                                               mediapipe::
                                                   NormalizedLandmarkList>>(),
                                       /* hand_world_landmarks */
                                       output_packets
                                           [kHandWorldLandmarksStreamName]
                                               .Get<std::vector<
                                                   mediapipe::LandmarkList>>());
}

absl::Status HandLandmarker::DetectAsync(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image,
                                              /*roi_allowed=*/false));
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kNormRectStreamName,
        MakePacket<NormalizedRect>(std::move(norm_rect))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace hand_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
