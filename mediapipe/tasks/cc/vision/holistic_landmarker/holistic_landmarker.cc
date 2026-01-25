/* Copyright 2024 The MediaPipe Authors.

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

#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker.h"

#include <memory>
#include <vector>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/base_task_api.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using HolisticLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    holistic_landmarker::proto::HolisticLandmarkerGraphOptions;

constexpr char kHolisticLandmarkerGraphTypeName[] =
    "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kFaceLandmarksTag[] = "FACE_LANDMARKS";
constexpr char kFaceLandmarksStreamName[] = "face_landmarks";
constexpr char kPoseLandmarksTag[] = "POSE_LANDMARKS";
constexpr char kPoseLandmarksStreamName[] = "pose_landmarks";
constexpr char kPoseWorldLandmarksTag[] = "POSE_WORLD_LANDMARKS";
constexpr char kPoseWorldLandmarksStreamName[] = "pose_world_landmarks";
constexpr char kPoseSegmentationMaskTag[] = "POSE_SEGMENTATION_MASK";
constexpr char kPoseSegmentationMaskStreamName[] = "pose_segmentation_mask";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<HolisticLandmarkerGraphOptionsProto> options,
    bool enable_flow_limiting, bool output_segmentation_masks) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHolisticLandmarkerGraphTypeName);
  subgraph.GetOptions<HolisticLandmarkerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  subgraph.Out(kPoseLandmarksTag).SetName(kPoseLandmarksStreamName) >>
      graph.Out(kPoseLandmarksTag);
  subgraph.Out(kPoseWorldLandmarksTag).SetName(kPoseWorldLandmarksStreamName) >>
      graph.Out(kPoseWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (output_segmentation_masks) {
    subgraph.Out(kPoseSegmentationMaskTag).SetName(kPoseSegmentationMaskStreamName) >>
        graph.Out(kPoseSegmentationMaskTag);
  }
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, subgraph, {kImageTag, kNormRectTag}, kNormLandmarksTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  return graph.GetConfig();
}

// Converts the user-facing HolisticLandmarkerOptions struct to the internal
// HolisticLandmarkerGraphOptions proto.
std::unique_ptr<HolisticLandmarkerGraphOptionsProto>
ConvertHolisticLandmarkerGraphOptionsProto(HolisticLandmarkerOptions* options) {
  auto options_proto = std::make_unique<HolisticLandmarkerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure face detector and face landmarks detector options.
  auto* face_detector_graph_options =
      options_proto->mutable_face_detector_graph_options();
  face_detector_graph_options->set_min_detection_confidence(
      options->min_face_detection_confidence);
  face_detector_graph_options->set_min_suppression_threshold(
      options->min_face_suppression_threshold);

  auto* face_landmarks_graph_options =
      options_proto->mutable_face_landmarks_detector_graph_options();
  face_landmarks_graph_options->set_min_detection_confidence(
      options->min_face_landmarks_confidence);

  // Configure pose detector and pose landmarks detector options.
  auto* pose_detector_graph_options =
      options_proto->mutable_pose_detector_graph_options();
  pose_detector_graph_options->set_min_detection_confidence(
      options->min_pose_detection_confidence);
  pose_detector_graph_options->set_min_suppression_threshold(
      options->min_pose_suppression_threshold);

  auto* pose_landmarks_graph_options =
      options_proto->mutable_pose_landmarks_detector_graph_options();
  pose_landmarks_graph_options->set_min_detection_confidence(
      options->min_pose_landmarks_confidence);

  // Configure hand landmarks detector options.
  auto* hand_landmarks_graph_options =
       options_proto->mutable_hand_landmarks_detector_graph_options();
  hand_landmarks_graph_options->set_min_detection_confidence(
        options->min_hand_landmarks_confidence);

  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HolisticLandmarker>> HolisticLandmarker::Create(
    std::unique_ptr<HolisticLandmarkerOptions> options) {
  auto options_proto = ConvertHolisticLandmarkerGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    bool output_segmentation_masks = options->output_segmentation_mask;
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
      if (status_or_packets.value()[kFaceLandmarksStreamName].IsEmpty()) {
        Packet empty_packet =
            status_or_packets.value()[kFaceLandmarksStreamName];
        result_callback(
            {HolisticLandmarkerResult()}, image_packet.Get<Image>(),
            empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        return;
      }
      Packet segmentation_mask_packet =
          status_or_packets.value()[kPoseSegmentationMaskStreamName];
      Packet pose_landmarks_packet =
          status_or_packets.value()[kPoseLandmarksStreamName];
      Packet pose_world_landmarks_packet =
          status_or_packets.value()[kPoseWorldLandmarksStreamName];
      std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
      if (output_segmentation_mask) {
        segmentation_mask = segmentation_mask_packet.Get<std::vector<Image>>();
      }
      result_callback(
          ConvertToHolisticLandmarkerResult(
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
      std::unique_ptr<HolisticLandmarker> holistic_landmarker,
      (core::VisionTaskApiFactory::Create<HolisticLandmarker,
                                          HolisticLandmarkerGraphOptionsProto>(
          CreateGraphConfig(
              std::move(options_proto),
              options->running_mode == core::RunningMode::LIVE_STREAM,
              options->output_segmentation_masks),
          std::move(options->base_options.op_resolver), options->running_mode,
          std::move(packets_callback))));

  holistic_landmarker->output_segmentation_mask_ =
      options->output_segmentation_mask;

  return holistic_landmarker;
}

absl::StatusOr<HolisticLandmarkerResult> HolisticLandmarker::Detect(
    mediapipe::Image image) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "GPU input images are currently not supported.",
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))}}));
  if (output_packets[kFaceLandmarksStreamName].IsEmpty()) {
    return {HolisticLandmarkerResult()};
  }
  std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
  if (output_segmentation_mask_) {
    segmentation_mask =
        output_packets[kPoseSegmentationMaskStreamName].Get<std::vector<Image>>();
  }
  return ConvertToHolisticLandmarkerResult(
      /* segmentation_mask= */
      segmentation_mask,
      /* pose_landmarks= */
      output_packets[kPoseLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* pose_world_landmarks */
      output_packets[kPoseWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
}

absl::StatusOr<HolisticLandmarkerResult> HolisticLandmarker::DetectForVideo(
    mediapipe::Image image, int64_t timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}
           }));
  if (output_packets[kFaceLandmarksStreamName].IsEmpty()) {
    return {HolisticLandmarkerResult()};
  }
  std::optional<std::vector<Image>> segmentation_mask = std::nullopt;
  if (output_segmentation_mask_) {
    segmentation_mask =
        output_packets[kPoseSegmentationMaskStreamName].Get<std::vector<Image>>();
  }
  return ConvertToHolisticLandmarkerResult(
      /* segmentation_mask= */
      segmentation_mask,
      /* pose_landmarks= */
      output_packets[kPoseLandmarksStreamName]
          .Get<std::vector<mediapipe::NormalizedLandmarkList>>(),
      /* pose_world_landmarks */
      output_packets[kPoseWorldLandmarksStreamName]
          .Get<std::vector<mediapipe::LandmarkList>>());
}

absl::Status HolisticLandmarker::DetectAsync(
    mediapipe::Image image, int64_t timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       });
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
