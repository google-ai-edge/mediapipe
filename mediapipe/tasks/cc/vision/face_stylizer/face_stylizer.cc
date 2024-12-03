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

#include "mediapipe/tasks/cc/vision/face_stylizer/face_stylizer.h"

#include <stdint.h>

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/face_stylizer/proto/face_stylizer_graph_options.pb.h"
#include "tensorflow/lite/core/api/op_resolver.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_stylizer {
namespace {

constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectName[] = "norm_rect_in";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph";
constexpr char kStylizedImageTag[] = "STYLIZED_IMAGE";
constexpr char kStylizedImageName[] = "stylized_image";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using FaceStylizerGraphOptionsProto =
    ::mediapipe::tasks::vision::face_stylizer::proto::FaceStylizerGraphOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<FaceStylizerGraphOptionsProto> options,
    bool enable_flow_limiting = false) {
  api2::builder::Graph graph;
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<FaceStylizerGraphOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectName);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  task_subgraph.Out(kStylizedImageTag).SetName(kStylizedImageName) >>
      graph.Out(kStylizedImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_subgraph, {kImageTag, kNormRectTag}, kStylizedImageTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> task_subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing FaceStylizerOptions struct to the internal
// FaceStylizerGraphOptions proto.
std::unique_ptr<FaceStylizerGraphOptionsProto>
ConvertFaceStylizerOptionsToProto(FaceStylizerOptions* options) {
  auto options_proto = std::make_unique<FaceStylizerGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<FaceStylizer>> FaceStylizer::Create(
    std::unique_ptr<FaceStylizerOptions> options) {
  auto options_proto = ConvertFaceStylizerOptionsToProto(options.get());
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
          Packet stylized_image_packet =
              status_or_packets.value()[kStylizedImageName];
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          result_callback(
              stylized_image_packet.IsEmpty()
                  ? std::nullopt
                  : std::optional<Image>(stylized_image_packet.Get<Image>()),
              image_packet.Get<Image>(),
              stylized_image_packet.Timestamp().Value() /
                  kMicroSecondsPerMilliSecond);
        };
  }
  return core::VisionTaskApiFactory::Create<FaceStylizer,
                                            FaceStylizerGraphOptionsProto>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver), core::RunningMode::IMAGE,
      std::move(packets_callback),
      /*disable_default_service=*/
      options->base_options.disable_default_service);
}

absl::StatusOr<std::optional<Image>> FaceStylizer::Stylize(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(image_processing_options, image));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, MakePacket<Image>(std::move(image))},
           {kNormRectName, MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  return output_packets[kStylizedImageName].IsEmpty()
             ? std::nullopt
             : std::optional<Image>(
                   output_packets[kStylizedImageName].Get<Image>());
}

}  // namespace face_stylizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
