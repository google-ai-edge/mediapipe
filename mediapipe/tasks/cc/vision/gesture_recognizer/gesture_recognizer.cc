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

#include "mediapipe/tasks/cc/vision/gesture_recognizer/gesture_recognizer.h"

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/packet.h"
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
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace gesture_recognizer {

namespace {

using GestureRecognizerGraphOptionsProto = ::mediapipe::tasks::vision::
    gesture_recognizer::proto::GestureRecognizerGraphOptions;

using ::mediapipe::NormalizedRect;

constexpr char kHandGestureSubgraphTypeName[] =
    "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph";

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kHandGesturesTag[] = "HAND_GESTURES";
constexpr char kHandGesturesStreamName[] = "hand_gestures";
constexpr char kHandednessTag[] = "HANDEDNESS";
constexpr char kHandednessStreamName[] = "handedness";
constexpr char kHandLandmarksTag[] = "LANDMARKS";
constexpr char kHandLandmarksStreamName[] = "landmarks";
constexpr char kHandWorldLandmarksTag[] = "WORLD_LANDMARKS";
constexpr char kHandWorldLandmarksStreamName[] = "world_landmarks";
constexpr int kMicroSecondsPerMilliSecond = 1000;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.GestureRecognizerGraph". If the task is running
// in the live stream mode, a "FlowLimiterCalculator" will be added to limit the
// number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<GestureRecognizerGraphOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kHandGestureSubgraphTypeName);
  subgraph.GetOptions<GestureRecognizerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  subgraph.Out(kHandGesturesTag).SetName(kHandGesturesStreamName) >>
      graph.Out(kHandGesturesTag);
  subgraph.Out(kHandednessTag).SetName(kHandednessStreamName) >>
      graph.Out(kHandednessTag);
  subgraph.Out(kHandLandmarksTag).SetName(kHandLandmarksStreamName) >>
      graph.Out(kHandLandmarksTag);
  subgraph.Out(kHandWorldLandmarksTag).SetName(kHandWorldLandmarksStreamName) >>
      graph.Out(kHandWorldLandmarksTag);
  subgraph.Out(kImageTag).SetName(kImageOutStreamName) >> graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, subgraph, {kImageTag, kNormRectTag}, kHandGesturesTag);
  }
  graph.In(kImageTag) >> subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing GestureRecognizerOptions struct to the internal
// GestureRecognizerGraphOptions proto.
std::unique_ptr<GestureRecognizerGraphOptionsProto>
ConvertGestureRecognizerGraphOptionsProto(GestureRecognizerOptions* options) {
  auto options_proto = std::make_unique<GestureRecognizerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);

  // Configure hand detector options.
  auto* hand_detector_graph_options =
      options_proto->mutable_hand_landmarker_graph_options()
          ->mutable_hand_detector_graph_options();
  hand_detector_graph_options->set_num_hands(options->num_hands);
  hand_detector_graph_options->set_min_detection_confidence(
      options->min_hand_detection_confidence);

  // Configure hand landmark detector options.
  auto* hand_landmarker_graph_options =
      options_proto->mutable_hand_landmarker_graph_options();
  hand_landmarker_graph_options->set_min_tracking_confidence(
      options->min_tracking_confidence);
  auto* hand_landmarks_detector_graph_options =
      hand_landmarker_graph_options
          ->mutable_hand_landmarks_detector_graph_options();
  hand_landmarks_detector_graph_options->set_min_detection_confidence(
      options->min_hand_presence_confidence);

  // Configure hand gesture recognizer options.
  auto* hand_gesture_recognizer_graph_options =
      options_proto->mutable_hand_gesture_recognizer_graph_options();
  auto canned_gestures_classifier_options_proto =
      std::make_unique<components::processors::proto::ClassifierOptions>(
          components::processors::ConvertClassifierOptionsToProto(
              &(options->canned_gestures_classifier_options)));
  hand_gesture_recognizer_graph_options
      ->mutable_canned_gesture_classifier_graph_options()
      ->mutable_classifier_options()
      ->Swap(canned_gestures_classifier_options_proto.get());
  auto custom_gestures_classifier_options_proto =
      std::make_unique<components::processors::proto::ClassifierOptions>(
          components::processors::ConvertClassifierOptionsToProto(
              &(options->custom_gestures_classifier_options)));
  hand_gesture_recognizer_graph_options
      ->mutable_custom_gesture_classifier_graph_options()
      ->mutable_classifier_options()
      ->Swap(custom_gestures_classifier_options_proto.get());
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<GestureRecognizer>> GestureRecognizer::Create(
    std::unique_ptr<GestureRecognizerOptions> options) {
  auto options_proto = ConvertGestureRecognizerGraphOptionsProto(options.get());
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
      if (status_or_packets.value()[kHandGesturesStreamName].IsEmpty()) {
        Packet empty_packet =
            status_or_packets.value()[kHandGesturesStreamName];
        result_callback(
            {{{}, {}, {}, {}}}, image_packet.Get<Image>(),
            empty_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        return;
      }
      Packet gesture_packet =
          status_or_packets.value()[kHandGesturesStreamName];
      Packet handedness_packet =
          status_or_packets.value()[kHandednessStreamName];
      Packet hand_landmarks_packet =
          status_or_packets.value()[kHandLandmarksStreamName];
      Packet hand_world_landmarks_packet =
          status_or_packets.value()[kHandWorldLandmarksStreamName];
      result_callback(
          {{gesture_packet.Get<std::vector<ClassificationList>>(),
            handedness_packet.Get<std::vector<ClassificationList>>(),
            hand_landmarks_packet.Get<std::vector<NormalizedLandmarkList>>(),
            hand_world_landmarks_packet.Get<std::vector<LandmarkList>>()}},
          image_packet.Get<Image>(),
          gesture_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
    };
  }
  return core::VisionTaskApiFactory::Create<GestureRecognizer,
                                            GestureRecognizerGraphOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<GestureRecognizerResult> GestureRecognizer::Recognize(
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
  if (output_packets[kHandGesturesStreamName].IsEmpty()) {
    return {{{}, {}, {}, {}}};
  }
  return {
      {/* gestures= */ {output_packets[kHandGesturesStreamName]
                            .Get<std::vector<ClassificationList>>()},
       /* handedness= */
       {output_packets[kHandednessStreamName]
            .Get<std::vector<mediapipe::ClassificationList>>()},
       /* hand_landmarks= */
       {output_packets[kHandLandmarksStreamName]
            .Get<std::vector<mediapipe::NormalizedLandmarkList>>()},
       /* hand_world_landmarks */
       {output_packets[kHandWorldLandmarksStreamName]
            .Get<std::vector<mediapipe::LandmarkList>>()}},
  };
}

absl::StatusOr<GestureRecognizerResult> GestureRecognizer::RecognizeForVideo(
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
  if (output_packets[kHandGesturesStreamName].IsEmpty()) {
    return {{{}, {}, {}, {}}};
  }
  return {
      {/* gestures= */ {output_packets[kHandGesturesStreamName]
                            .Get<std::vector<ClassificationList>>()},
       /* handedness= */
       {output_packets[kHandednessStreamName]
            .Get<std::vector<mediapipe::ClassificationList>>()},
       /* hand_landmarks= */
       {output_packets[kHandLandmarksStreamName]
            .Get<std::vector<mediapipe::NormalizedLandmarkList>>()},
       /* hand_world_landmarks */
       {output_packets[kHandWorldLandmarksStreamName]
            .Get<std::vector<mediapipe::LandmarkList>>()}},
  };
}

absl::Status GestureRecognizer::RecognizeAsync(
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

}  // namespace gesture_recognizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
