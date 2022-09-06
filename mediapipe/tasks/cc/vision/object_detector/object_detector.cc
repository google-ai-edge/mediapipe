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

#include "mediapipe/tasks/cc/vision/object_detector/object_detector.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/object_detector/proto/object_detector_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

constexpr char kDetectionsOutStreamName[] = "detections_out";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.ObjectDetectorGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ObjectDetectorOptionsProto =
    object_detector::proto::ObjectDetectorOptions;

// Creates a MediaPipe graph config that contains a subgraph node of
// "mediapipe.tasks.vision.ObjectDetectorGraph". If the task is running in the
// live stream mode, a "FlowLimiterCalculator" will be added to limit the
// number of frames in flight.
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ObjectDetectorOptionsProto> options_proto,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  graph.In(kImageTag).SetName(kImageInStreamName);
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<ObjectDetectorOptionsProto>().Swap(
      options_proto.get());
  task_subgraph.Out(kDetectionsTag).SetName(kDetectionsOutStreamName) >>
      graph.Out(kDetectionsTag);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(graph, task_subgraph,
                                                 {kImageTag}, kDetectionsTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  return graph.GetConfig();
}

// Converts the user-facing ObjectDetectorOptions struct to the internal
// ObjectDetectorOptions proto.
std::unique_ptr<ObjectDetectorOptionsProto> ConvertObjectDetectorOptionsToProto(
    ObjectDetectorOptions* options) {
  auto options_proto = std::make_unique<ObjectDetectorOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  options_proto->set_display_names_locale(options->display_names_locale);
  options_proto->set_max_results(options->max_results);
  options_proto->set_score_threshold(options->score_threshold);
  for (const std::string& category : options->category_allowlist) {
    options_proto->add_category_allowlist(category);
  }
  for (const std::string& category : options->category_denylist) {
    options_proto->add_category_denylist(category);
  }
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ObjectDetector>> ObjectDetector::Create(
    std::unique_ptr<ObjectDetectorOptions> options) {
  auto options_proto = ConvertObjectDetectorOptionsToProto(options.get());
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
          Packet detections_packet =
              status_or_packets.value()[kDetectionsOutStreamName];
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          result_callback(detections_packet.Get<std::vector<Detection>>(),
                          image_packet.Get<Image>(),
                          detections_packet.Timestamp().Value() /
                              kMicroSecondsPerMilliSecond);
        };
  }
  return core::VisionTaskApiFactory::Create<ObjectDetector,
                                            ObjectDetectorOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<std::vector<Detection>> ObjectDetector::Detect(
    mediapipe::Image image) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(auto output_packets,
                   ProcessImageData({{kImageInStreamName,
                                      MakePacket<Image>(std::move(image))}}));
  return output_packets[kDetectionsOutStreamName].Get<std::vector<Detection>>();
}

absl::StatusOr<std::vector<Detection>> ObjectDetector::Detect(
    mediapipe::Image image, int64 timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
  return output_packets[kDetectionsOutStreamName].Get<std::vector<Detection>>();
}

absl::Status ObjectDetector::DetectAsync(Image image, int64 timestamp_ms) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
