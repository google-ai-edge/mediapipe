/* Copyright 2026 The MediaPipe Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/task_runner.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/holistic_landmarker_result.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_detector/proto/pose_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/pose_landmarker/proto/pose_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace holistic_landmarker {

namespace {

using HolisticLandmarkerGraphOptionsProto = ::mediapipe::tasks::vision::
    holistic_landmarker::proto::HolisticLandmarkerGraphOptions;

using ::mediapipe::tasks::components::containers::Category;
using ::mediapipe::tasks::components::containers::ConvertToCategory;
using ::mediapipe::tasks::components::containers::ConvertToLandmarks;
using ::mediapipe::tasks::components::containers::ConvertToNormalizedLandmarks;

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
constexpr char kLeftHandLandmarksTag[] = "LEFT_HAND_LANDMARKS";
constexpr char kLeftHandLandmarksStreamName[] = "left_hand_landmarks";
constexpr char kRightHandLandmarksTag[] = "RIGHT_HAND_LANDMARKS";
constexpr char kRightHandLandmarksStreamName[] = "right_hand_landmarks";
constexpr char kLeftHandWorldLandmarksTag[] = "LEFT_HAND_WORLD_LANDMARKS";
constexpr char kLeftHandWorldLandmarksStreamName[] =
    "left_hand_world_landmarks";
constexpr char kRightHandWorldLandmarksTag[] = "RIGHT_HAND_WORLD_LANDMARKS";
constexpr char kRightHandWorldLandmarksStreamName[] =
    "right_hand_world_landmarks";
constexpr char kPoseSegmentationMaskTag[] = "POSE_SEGMENTATION_MASK";
constexpr char kPoseSegmentationMaskStreamName[] = "pose_segmentation_mask";
constexpr char kFaceBlendshapesTag[] = "FACE_BLENDSHAPES";
constexpr char kFaceBlendshapesStreamName[] = "face_blendshapes";

constexpr int kMicroSecondsPerMilliSecond = 1000;

HolisticLandmarkerResult ConvertToHolisticLandmarkerResult(
    const tasks::core::PacketMap& packets, bool output_face_blendshapes,
    bool output_pose_segmentation_masks) {
  HolisticLandmarkerResult result;
  NormalizedLandmarkList face_landmarks_proto;
  if (!packets.at(kFaceLandmarksStreamName).IsEmpty()) {
    face_landmarks_proto =
        packets.at(kFaceLandmarksStreamName).Get<NormalizedLandmarkList>();
  }
  result.face_landmarks = ConvertToNormalizedLandmarks(face_landmarks_proto);

  NormalizedLandmarkList pose_landmarks_proto;
  if (!packets.at(kPoseLandmarksStreamName).IsEmpty()) {
    pose_landmarks_proto =
        packets.at(kPoseLandmarksStreamName).Get<NormalizedLandmarkList>();
  }
  result.pose_landmarks = ConvertToNormalizedLandmarks(pose_landmarks_proto);

  LandmarkList pose_world_landmarks_proto;
  if (!packets.at(kPoseWorldLandmarksStreamName).IsEmpty()) {
    pose_world_landmarks_proto =
        packets.at(kPoseWorldLandmarksStreamName).Get<LandmarkList>();
  }
  result.pose_world_landmarks = ConvertToLandmarks(pose_world_landmarks_proto);

  NormalizedLandmarkList left_hand_landmarks_proto;
  if (!packets.at(kLeftHandLandmarksStreamName).IsEmpty()) {
    left_hand_landmarks_proto =
        packets.at(kLeftHandLandmarksStreamName).Get<NormalizedLandmarkList>();
  }
  result.left_hand_landmarks =
      ConvertToNormalizedLandmarks(left_hand_landmarks_proto);

  NormalizedLandmarkList right_hand_landmarks_proto;
  if (!packets.at(kRightHandLandmarksStreamName).IsEmpty()) {
    right_hand_landmarks_proto =
        packets.at(kRightHandLandmarksStreamName).Get<NormalizedLandmarkList>();
  }
  result.right_hand_landmarks =
      ConvertToNormalizedLandmarks(right_hand_landmarks_proto);

  LandmarkList left_hand_world_landmarks_proto;
  if (!packets.at(kLeftHandWorldLandmarksStreamName).IsEmpty()) {
    left_hand_world_landmarks_proto =
        packets.at(kLeftHandWorldLandmarksStreamName).Get<LandmarkList>();
  }
  result.left_hand_world_landmarks =
      ConvertToLandmarks(left_hand_world_landmarks_proto);

  LandmarkList right_hand_world_landmarks_proto;
  if (!packets.at(kRightHandWorldLandmarksStreamName).IsEmpty()) {
    right_hand_world_landmarks_proto =
        packets.at(kRightHandWorldLandmarksStreamName).Get<LandmarkList>();
  }
  result.right_hand_world_landmarks =
      ConvertToLandmarks(right_hand_world_landmarks_proto);

  if (output_pose_segmentation_masks &&
      packets.count(kPoseSegmentationMaskStreamName) &&
      !packets.at(kPoseSegmentationMaskStreamName).IsEmpty()) {
    result.pose_segmentation_masks =
        packets.at(kPoseSegmentationMaskStreamName).Get<Image>();
  }

  if (output_face_blendshapes && packets.count(kFaceBlendshapesStreamName) &&
      !packets.at(kFaceBlendshapesStreamName).IsEmpty()) {
    ClassificationList face_blendshapes_proto =
        packets.at(kFaceBlendshapesStreamName).Get<ClassificationList>();
    std::vector<Category> face_blendshapes_categories;
    for (const auto& classification : face_blendshapes_proto.classification()) {
      face_blendshapes_categories.push_back(ConvertToCategory(classification));
    }
    if (!face_blendshapes_categories.empty()) {
      result.face_blendshapes = face_blendshapes_categories;
    }
  }
  return result;
}

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.holistic_ladnamrker.HolisticLandmarkerGraph". If the
// task is running in the live stream mode, a "FlowLimiterCalculator" will be
// added to limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<HolisticLandmarkerGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHolisticLandmarkerGraphTypeName);
  subgraph.GetOptions<HolisticLandmarkerGraphOptionsProto>().Swap(
      options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  subgraph.Out(kFaceLandmarksTag).SetName(kFaceLandmarksStreamName) >>
      graph.Out(kFaceLandmarksTag);
  subgraph.Out(kPoseLandmarksTag).SetName(kPoseLandmarksStreamName) >>
      graph.Out(kPoseLandmarksTag);
  subgraph.Out(kPoseWorldLandmarksTag).SetName(kPoseWorldLandmarksStreamName) >>
      graph.Out(kPoseWorldLandmarksTag);
  subgraph.Out(kLeftHandLandmarksTag).SetName(kLeftHandLandmarksStreamName) >>
      graph.Out(kLeftHandLandmarksTag);
  subgraph.Out(kRightHandLandmarksTag).SetName(kRightHandLandmarksStreamName) >>
      graph.Out(kRightHandLandmarksTag);
  subgraph.Out(kLeftHandWorldLandmarksTag)
          .SetName(kLeftHandWorldLandmarksStreamName) >>
      graph.Out(kLeftHandWorldLandmarksTag);
  subgraph.Out(kRightHandWorldLandmarksTag)
          .SetName(kRightHandWorldLandmarksStreamName) >>
      graph.Out(kRightHandWorldLandmarksTag);
  subgraph.Out(kPoseSegmentationMaskTag)
          .SetName(kPoseSegmentationMaskStreamName) >>
      graph.Out(kPoseSegmentationMaskTag);
  subgraph.Out(kFaceBlendshapesTag).SetName(kFaceBlendshapesStreamName) >>
      graph.Out(kFaceBlendshapesTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(graph, subgraph, {kImageTag},
                                                 kPoseLandmarksTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  return graph.GetConfig();
}

// Converts the user-facing HolisticLandmarkerOptions struct to the internal
// HolisticLandmarkerGraphOptions proto.
std::unique_ptr<HolisticLandmarkerGraphOptionsProto>
ConvertHolisticLandmarkerOptionsProto(HolisticLandmarkerOptions* options) {
  auto options_proto = std::make_unique<HolisticLandmarkerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure face detector options.
  options_proto->mutable_face_detector_graph_options()
      ->set_min_detection_confidence(options->min_face_detection_confidence);
  options_proto->mutable_face_detector_graph_options()
      ->set_min_suppression_threshold(options->min_face_suppression_threshold);
  options_proto->mutable_face_landmarks_detector_graph_options()
      ->set_min_detection_confidence(options->min_face_presence_confidence);

  // Configure hand detector options.
  options_proto->mutable_hand_landmarks_detector_graph_options()
      ->set_min_detection_confidence(options->min_hand_landmarks_confidence);

  // Configure pose detector options.
  options_proto->mutable_pose_detector_graph_options()
      ->set_min_detection_confidence(options->min_pose_detection_confidence);
  options_proto->mutable_pose_detector_graph_options()
      ->set_min_suppression_threshold(options->min_pose_suppression_threshold);
  options_proto->mutable_pose_landmarks_detector_graph_options()
      ->set_min_detection_confidence(options->min_pose_presence_confidence);

  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HolisticLandmarker>> HolisticLandmarker::Create(
    std::unique_ptr<HolisticLandmarkerOptions> options) {
  auto options_proto = ConvertHolisticLandmarkerOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    bool output_pose_segmentation_masks =
        options->output_pose_segmentation_masks;
    bool output_face_blendshapes = options->output_face_blendshapes;
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
      result_callback(
          ConvertToHolisticLandmarkerResult(status_or_packets.value(),
                                            output_face_blendshapes,
                                            output_pose_segmentation_masks),
          image_packet.Get<Image>(),
          status_or_packets.value()[kImageOutStreamName].Timestamp().Value() /
              kMicroSecondsPerMilliSecond);
    };
  }
  MP_ASSIGN_OR_RETURN(
      auto landmarker,
      (core::VisionTaskApiFactory::Create<HolisticLandmarker,
                                          HolisticLandmarkerGraphOptionsProto>(
          CreateGraphConfig(
              std::move(options_proto),
              options->running_mode == core::RunningMode::LIVE_STREAM),
          std::move(options->base_options.op_resolver), options->running_mode,
          std::move(packets_callback),
          /*disable_default_service=*/
          options->base_options.disable_default_service)));
  landmarker->output_pose_segmentation_masks_ =
      options->output_pose_segmentation_masks;
  landmarker->output_face_blendshapes_ = options->output_face_blendshapes;
  return landmarker;
}

absl::StatusOr<HolisticLandmarkerResult> HolisticLandmarker::Detect(
    mediapipe::Image image, const std::optional<core::ImageProcessingOptions>&
                                image_processing_options) {
  if (image_processing_options.has_value() &&
      image_processing_options->region_of_interest.has_value()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "This task doesn't support region-of-interest.",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))}}));
  return ConvertToHolisticLandmarkerResult(output_packets,
                                           output_face_blendshapes_,
                                           output_pose_segmentation_masks_);
}

absl::StatusOr<HolisticLandmarkerResult> HolisticLandmarker::DetectForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    const std::optional<core::ImageProcessingOptions>&
        image_processing_options) {
  if (image_processing_options.has_value() &&
      image_processing_options->region_of_interest.has_value()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "This task doesn't support region-of-interest.",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return ConvertToHolisticLandmarkerResult(output_packets,
                                           output_face_blendshapes_,
                                           output_pose_segmentation_masks_);
}

absl::Status HolisticLandmarker::DetectAsync(
    mediapipe::Image image, int64_t timestamp_ms,
    const std::optional<core::ImageProcessingOptions>&
        image_processing_options) {
  if (image_processing_options.has_value() &&
      image_processing_options->region_of_interest.has_value()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "This task doesn't support region-of-interest.",
        MediaPipeTasksStatus::kImageProcessingInvalidArgumentError);
  }
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace holistic_landmarker
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
