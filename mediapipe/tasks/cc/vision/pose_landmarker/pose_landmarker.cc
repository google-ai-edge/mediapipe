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

#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker.h"

#include <memory>
#include <utility>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/pose_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace pose_landmarker {

namespace {

using PoseLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    pose_landmarker::proto::PoseLandmarkerGraphOptions;

using ::mediapipe::NormalizedRect;

constexpr char kPoseLandmarkerGraphTypeName[] =
    "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kSegmentationMaskTag[] = "SEGMENTATION_MASK";
constexpr char kSegmentationMaskStreamName[] = "segmentation_mask";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kNormLandmarksStreamName[] = "norm_landmarks";
constexpr char kPoseWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kPoseWorldLandmarksStreamName[] = "world_landmarks";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.pose_ladnamrker.PoseLandmarkerGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<PoseLandmarkerGraphOptionsProto> options,
    bool enable_flow_limiting, bool output_segmentation_masks) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kPoseLandmarkerGraphTypeName);
  subgraph.GetOptions<PoseLandmarkerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  subgraph.Out(kNormLandmarksTag).SetName(kNormLandmarksStreamName) >>
      graph.Out(kNormLandmarksTag);
  subgraph.Out(kPoseWorldLandmarksTag).SetName(kPoseWorldLandmarksStreamName) >>
      graph.Out(kPoseWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (output_segmentation_masks) {
    subgraph.Out(kSegmentationMaskTag).SetName(kSegmentationMaskStreamName) >>
        graph.Out(kSegmentationMaskTag);
  }
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, subgraph, {kImageTag, kNormRectTag}, kNormLandmarksTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing PoseLandmarkerOptions struct to the internal
// PoseLandmarkerGraphOptions proto.
std::unique_ptr<PoseLandmarkerGraphOptionsProto>
ConvertPoseLandmarkerGraphOptionsProto(PoseLandmarkerOptions* options) {
  auto options_proto = std::make_unique<PoseLandmarkerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure pose detector options.
  auto* pose_detector_graph_options =
      options_proto->mutable_pose_detector_graph_options();
  pose_detector_graph_options->set_num_poses(options->num_poses);
  pose_detector_graph_options->set_min_detection_confidence(
      options->min_pose_detection_confidence);

  // Configure pose landmark detector options.
  options_proto->set_min_tracking_confidence(options->min_tracking_confidence);
  auto* pose_landmarks_detector_graph_options =
      options_proto->mutable_pose_landmarks_detector_graph_options();
  pose_landmarks_detector_graph_options->set_min_detection_confidence(
      options->min_pose_presence_confidence);

  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<PoseLandmarker>> PoseLandmarker::Create(
    std::unique_ptr<PoseLandmarkerOptions> options) {
  auto options_proto = ConvertPoseLandmarkerGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    bool output_segmentation_masks = options->output_segmentation_masks;
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
      if (status_or_packets.value()[kNormLandmarksStreamName].IsEmpty()) {
        Packet empty_packet =
            status_or_packets.value()[kNormLandmarksStreamName];
        result_callback(
            {PoseLandmarkerResult()}, image_packet.Get<Image>(),
            empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        return;
      }
      Packet segmentation_mask_packet =
          status_or_packets.value()[kSegmentationMaskStreamName];
      Packet pose_landmarks_packet =
          status_or_packets.value()[kNormLandmarksStreamName];
      Packet pose_world_landmarks_packet =
          status_or_packets.value()[kPoseWorldLandmarksStreamName];
      std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
      if (output_segmentation_masks) {
        segmentation_mask = segmentation_mask_packet.Get<std::vector<Image>>();
      }
      result_callback(
          ConvertToPoseLandmarkerResult(
              /* segmentation_mask= */ segmentation_mask,
              /* pose_landmarks= */
              pose_landmarks_packet.Get<std::vector<NormalizedLandmarkList>>(),
              /* pose_world_landmarks= */
              pose_world_landmarks_packet.Get<std::vector<LandmarkList>>()),
          image_packet.Get<Image>(),
          pose_landmarks_packet.Timestamp().Value() /
              kMicroSecondsPerMilliSecond);
    };
  }
  MP_ASSIGN_OR_RETURN(
      std::unique_ptr<PoseLandmarker> pose_landmarker,
      (core::VisionTaskApiFactory::Create<PoseLandmarker,
                                          PoseLandmarkerGraphOptionsProto>(
          CreateGraphConfig(
              std::move(options_proto),
              options->running_mode == core::RunningMode::LIVE_STREAM,
              options->output_segmentation_masks),
          std::move(options->base_options.op_resolver), options->running_mode,
          std::move(packets_callback),
          /*disable_default_service=*/
          options->base_options.disable_default_service)));

  pose_landmarker->output_segmentation_masks_ =
      options->output_segmentation_masks;

  return pose_landmarker;
}

absl::StatusOr<PoseLandmarkerResult> PoseLandmarker::Detect(
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
  if (output_packets[kNormLandmarksStreamName].IsEmpty()) {
    return {PoseLandmarkerResult()};
  }
  std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
  if (output_segmentation_masks_) {
    segmentation_mask =
        output_packets[kSegmentationMaskStreamName].Get<std::vector<Image>>();
  }
  return ConvertToPoseLandmarkerResult(
      /* segmentation_mask= */
      segmentation_mask,
      /* pose_landmarks= */
      output_packets[kNormLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* pose_world_landmarks */
      output_packets[kPoseWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
}

absl::StatusOr<PoseLandmarkerResult> PoseLandmarker::DetectForVideo(
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
  if (output_packets[kNormLandmarksStreamName].IsEmpty()) {
    return {PoseLandmarkerResult()};
  }
  std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
  if (output_segmentation_masks_) {
    segmentation_mask =
        output_packets[kSegmentationMaskStreamName].Get<std::vector<Image>>();
  }
  return ConvertToPoseLandmarkerResult(
      /* segmentation_mask= */
      segmentation_mask,
      /* pose_landmarks= */
      output_packets[kNormLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* pose_world_landmarks */
      output_packets[kPoseWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
}

absl::Status PoseLandmarker::DetectAsync(
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

}  // namespace pose_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
