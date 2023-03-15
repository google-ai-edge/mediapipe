/* Copyright 2023 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"

#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {
namespace {

constexpr char kSegmentationStreamName[] = "segmented_mask_out";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kRoiStreamName[] = "roi_in";
constexpr char kNormRectStreamName[] = "norm_rect_in";

constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kImageTag[] = "IMAGE";
constexpr char kRoiTag[] = "ROI";
constexpr char kNormRectTag[] = "NORM_RECT";

constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph";

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;
using ImageSegmenterGraphOptionsProto = ::mediapipe::tasks::vision::
    image_segmenter::proto::ImageSegmenterGraphOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageSegmenterGraphOptionsProto> options) {
  api2::builder::Graph graph;
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<ImageSegmenterGraphOptionsProto>().Swap(
      options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kRoiTag).SetName(kRoiStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  task_subgraph.Out(kGroupedSegmentationTag).SetName(kSegmentationStreamName) >>
      graph.Out(kGroupedSegmentationTag);
  task_subgraph.Out(kImageTag).SetName(kImageOutStreamName) >>
      graph.Out(kImageTag);
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  graph.In(kRoiTag) >> task_subgraph.In(kRoiTag);
  graph.In(kNormRectTag) >> task_subgraph.In(kNormRectTag);
  return graph.GetConfig();
}

// Converts the user-facing InteractiveSegmenterOptions struct to the internal
// ImageSegmenterOptions proto.
std::unique_ptr<ImageSegmenterGraphOptionsProto>
ConvertImageSegmenterOptionsToProto(InteractiveSegmenterOptions* options) {
  auto options_proto = std::make_unique<ImageSegmenterGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  switch (options->output_type) {
    case InteractiveSegmenterOptions::OutputType::CATEGORY_MASK:
      options_proto->mutable_segmenter_options()->set_output_type(
          SegmenterOptions::CATEGORY_MASK);
      break;
    case InteractiveSegmenterOptions::OutputType::CONFIDENCE_MASK:
      options_proto->mutable_segmenter_options()->set_output_type(
          SegmenterOptions::CONFIDENCE_MASK);
      break;
  }
  return options_proto;
}

// Converts the user-facing RegionOfInterest struct to the RenderData proto that
// is used in subgraph.
absl::StatusOr<RenderData> ConvertRoiToRenderData(const RegionOfInterest& roi) {
  RenderData result;
  switch (roi.format) {
    case RegionOfInterest::UNSPECIFIED:
      return absl::InvalidArgumentError(
          "RegionOfInterest format not specified");
    case RegionOfInterest::KEYPOINT:
      RET_CHECK(roi.keypoint.has_value());
      auto* annotation = result.add_render_annotations();
      annotation->mutable_color()->set_r(255);
      auto* point = annotation->mutable_point();
      point->set_normalized(true);
      point->set_x(roi.keypoint->x);
      point->set_y(roi.keypoint->y);
      return result;
  }
  return absl::UnimplementedError("Unrecognized format");
}

}  // namespace

absl::StatusOr<std::unique_ptr<InteractiveSegmenter>>
InteractiveSegmenter::Create(
    std::unique_ptr<InteractiveSegmenterOptions> options) {
  auto options_proto = ConvertImageSegmenterOptionsToProto(options.get());
  return core::VisionTaskApiFactory::Create<InteractiveSegmenter,
                                            ImageSegmenterGraphOptionsProto>(
      CreateGraphConfig(std::move(options_proto)),
      std::move(options->base_options.op_resolver), core::RunningMode::IMAGE,
      /*packets_callback=*/nullptr);
}

absl::StatusOr<std::vector<Image>> InteractiveSegmenter::Segment(
    mediapipe::Image image, const RegionOfInterest& roi,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  ASSIGN_OR_RETURN(
      NormalizedRect norm_rect,
      ConvertToNormalizedRect(image_processing_options, /*roi_allowed=*/false));
  ASSIGN_OR_RETURN(RenderData roi_as_render_data, ConvertRoiToRenderData(roi));
  ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, mediapipe::MakePacket<Image>(std::move(image))},
           {kRoiStreamName,
            mediapipe::MakePacket<RenderData>(std::move(roi_as_render_data))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  return output_packets[kSegmentationStreamName].Get<std::vector<Image>>();
}

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
