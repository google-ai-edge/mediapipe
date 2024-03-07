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

#include "mediapipe/tasks/cc/vision/face_detector/face_detector.h"

#include <utility>

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/tasks/cc/components/containers/detection_result.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/base_vision_task_api.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_detector {

namespace {

using FaceDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions;

constexpr char kFaceDetectorGraphTypeName[] =
    "mediapipe.tasks.vision.face_detector.FaceDetectorGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionsStreamName[] = "detections";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.face_detector.FaceDetectorGraph". If the task is
// running in the live stream mode, a "FlowLimiterCalculator" will be added to
// limit the number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<FaceDetectorGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kFaceDetectorGraphTypeName);
  subgraph.GetOptions<FaceDetectorGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  subgraph.Out(kDetectionsTag).SetName(kDetectionsStreamName) >>
      graph.Out(kDetectionsTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, subgraph, {kImageTag, kNormRectTag}, kDetectionsTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing FaceDetectorOptions struct to the internal
// FaceDetectorGraphOptions proto.
std::unique_ptr<FaceDetectorGraphOptionsProto>
ConvertFaceDetectorGraphOptionsProto(FaceDetectorOptions* options) {
  auto options_proto = std::make_unique<FaceDetectorGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  options_proto->set_min_detection_confidence(
      options->min_detection_confidence);
  options_proto->set_min_suppression_threshold(
      options->min_suppression_threshold);
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<FaceDetector>> FaceDetector::Create(
    std::unique_ptr<FaceDetectorOptions> options) {
  auto options_proto = ConvertFaceDetectorGraphOptionsProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    packets_callback =
        [=](absl::StatusOr<tasks::core::PacketMap> status_or_packets) {
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
          if (status_or_packets.value()[kDetectionsStreamName].IsEmpty()) {
            Packet empty_packet =
                status_or_packets.value()[kDetectionsStreamName];
            result_callback(
                {FaceDetectorResult()}, image_packet.Get<Image>(),
                empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
            return;
          }
          Packet detections_packet =
              status_or_packets.value()[kDetectionsStreamName];
          result_callback(
              components::containers::ConvertToDetectionResult(
                  detections_packet.Get<std::vector<mediapipe::Detection>>()),
              image_packet.Get<Image>(),
              detections_packet.Timestamp().Value() /
                  kMicroSecondsPerMilliSecond);
        };
  }
  return core::VisionTaskApiFactory::Create<FaceDetector,
                                            FaceDetectorGraphOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<FaceDetectorResult> FaceDetector::Detect(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image,
                                              /*roi_allowed=*/false));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  if (output_packets[kDetectionsStreamName].IsEmpty()) {
    return {FaceDetectorResult()};
  }
  return components::containers::ConvertToDetectionResult(
      output_packets[kDetectionsStreamName]
          .Get<std::vector<mediapipe::Detection>>());
}

absl::StatusOr<FaceDetectorResult> FaceDetector::DetectForVideo(
    mediapipe::Image image, uint64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
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
  if (output_packets[kDetectionsStreamName].IsEmpty()) {
    return {FaceDetectorResult()};
  }
  return components::containers::ConvertToDetectionResult(
      output_packets[kDetectionsStreamName]
          .Get<std::vector<mediapipe::Detection>>());
}

absl::Status FaceDetector::DetectAsync(
    mediapipe::Image image, uint64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
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

}  // namespace face_detector
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
