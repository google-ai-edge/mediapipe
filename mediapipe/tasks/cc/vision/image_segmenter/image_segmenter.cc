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

#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/components/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

constexpr char kSegmentationStreamName[] = "segmented_mask_out";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.ImageSegmenterGraph";

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Image;
using ImageSegmenterOptionsProto =
    image_segmenter::proto::ImageSegmenterOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.ImageSegmenterGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageSegmenterOptionsProto> options,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<ImageSegmenterOptionsProto>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  task_subgraph.Out(kGroupedSegmentationTag).SetName(kSegmentationStreamName) >>
      graph.Out(kGroupedSegmentationTag);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_subgraph, {kImageTag}, kGroupedSegmentationTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  return graph.GetConfig();
}

// Converts the user-facing ImageSegmenterOptions struct to the internal
// ImageSegmenterOptions proto.
std::unique_ptr<ImageSegmenterOptionsProto> ConvertImageSegmenterOptionsToProto(
    ImageSegmenterOptions* options) {
  auto options_proto = std::make_unique<ImageSegmenterOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  options_proto->set_display_names_locale(options->display_names_locale);
  switch (options->output_type) {
    case ImageSegmenterOptions::OutputType::CATEGORY_MASK:
      options_proto->mutable_segmenter_options()->set_output_type(
          SegmenterOptions::CATEGORY_MASK);
      break;
    case ImageSegmenterOptions::OutputType::CONFIDENCE_MASK:
      options_proto->mutable_segmenter_options()->set_output_type(
          SegmenterOptions::CONFIDENCE_MASK);
      break;
  }
  switch (options->activation) {
    case ImageSegmenterOptions::Activation::NONE:
      options_proto->mutable_segmenter_options()->set_activation(
          SegmenterOptions::NONE);
      break;
    case ImageSegmenterOptions::Activation::SIGMOID:
      options_proto->mutable_segmenter_options()->set_activation(
          SegmenterOptions::SIGMOID);
      break;
    case ImageSegmenterOptions::Activation::SOFTMAX:
      options_proto->mutable_segmenter_options()->set_activation(
          SegmenterOptions::SOFTMAX);
      break;
  }
  return options_proto;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageSegmenter>> ImageSegmenter::Create(
    std::unique_ptr<ImageSegmenterOptions> options) {
  auto options_proto = ConvertImageSegmenterOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  return core::VisionTaskApiFactory::Create<ImageSegmenter,
                                            ImageSegmenterOptionsProto>(
      CreateGraphConfig(
          std::move(options_proto),
          options->running_mode == core::RunningMode::LIVE_STREAM),
      std::move(options->base_options.op_resolver), options->running_mode,
      std::move(packets_callback));
}

absl::StatusOr<std::vector<Image>> ImageSegmenter::Segment(
    mediapipe::Image image) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData({{kImageInStreamName,
                         mediapipe::MakePacket<Image>(std::move(image))}}));
  return output_packets[kSegmentationStreamName].Get<std::vector<Image>>();
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
