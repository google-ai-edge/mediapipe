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

#include "mediapipe/tasks/cc/vision/interactive_segmenter/interactive_segmenter.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/keypoint.h"
#include "mediapipe/tasks/cc/core/base_options.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter_result.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {
namespace {

constexpr char kConfidenceMasksStreamName[] = "confidence_masks";
constexpr char kCategoryMaskStreamName[] = "category_mask";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kRoiStreamName[] = "roi_in";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kQualityScoresStreamName[] = "quality_scores";

constexpr absl::string_view kConfidenceMasksTag{"CONFIDENCE_MASKS"};
constexpr absl::string_view kCategoryMaskTag{"CATEGORY_MASK"};
constexpr absl::string_view kImageTag{"IMAGE"};
constexpr absl::string_view kRoiTag{"ROI"};
constexpr absl::string_view kNormRectTag{"NORM_RECT"};
constexpr absl::string_view kQualityScoresTag{"QUALITY_SCORES"};

constexpr absl::string_view kSubgraphTypeName{
    "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph"};

using components::containers::NormalizedKeypoint;

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterResult;
using ImageSegmenterGraphOptionsProto = ::mediapipe::tasks::vision::
    image_segmenter::proto::ImageSegmenterGraphOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageSegmenterGraphOptionsProto> options,
    bool output_confidence_masks, bool output_category_mask) {
  api2::builder::Graph graph;
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<ImageSegmenterGraphOptionsProto>().Swap(
      options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kRoiTag).SetName(kRoiStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  if (output_confidence_masks) {
    task_subgraph.Out(kConfidenceMasksTag)
            .SetName(kConfidenceMasksStreamName) >>
        graph.Out(kConfidenceMasksTag);
  }
  if (output_category_mask) {
    task_subgraph.Out(kCategoryMaskTag).SetName(kCategoryMaskStreamName) >>
        graph.Out(kCategoryMaskTag);
  }
  task_subgraph.Out(kQualityScoresTag).SetName(kQualityScoresStreamName) >>
      graph.Out(kQualityScoresTag);
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
  return options_proto;
}

// Converts the user-facing RegionOfInterest struct to the RenderData proto that
// is used in subgraph.
absl::StatusOr<RenderData> ConvertRoiToRenderData(const RegionOfInterest& roi) {
  RenderData result;
  switch (roi.format) {
    case RegionOfInterest::Format::kUnspecified:
      return absl::InvalidArgumentError(
          "RegionOfInterest format not specified");
    case RegionOfInterest::Format::kKeyPoint: {
      RET_CHECK(roi.keypoint.has_value());
      auto* annotation = result.add_render_annotations();
      annotation->mutable_color()->set_r(255);
      auto* point = annotation->mutable_point();
      point->set_normalized(true);
      point->set_x(roi.keypoint->x);
      point->set_y(roi.keypoint->y);
      return result;
    }
    case RegionOfInterest::Format::kScribble: {
      RET_CHECK(roi.scribble.has_value());
      auto* annotation = result.add_render_annotations();
      annotation->mutable_color()->set_r(255);
      for (const NormalizedKeypoint& keypoint : *(roi.scribble)) {
        auto* point = annotation->mutable_scribble()->add_point();
        point->set_normalized(true);
        point->set_x(keypoint.x);
        point->set_y(keypoint.y);
      }
      return result;
    }
  }
  return absl::UnimplementedError("Unrecognized format");
}

}  // namespace

absl::StatusOr<std::unique_ptr<InteractiveSegmenter>>
InteractiveSegmenter::Create(
    std::unique_ptr<InteractiveSegmenterOptions> options) {
  if (!options->output_confidence_masks && !options->output_category_mask) {
    return absl::InvalidArgumentError(
        "At least one of `output_confidence_masks` and `output_category_mask` "
        "must be set.");
  }
  std::unique_ptr<ImageSegmenterGraphOptionsProto> options_proto =
      ConvertImageSegmenterOptionsToProto(options.get());
  MP_ASSIGN_OR_RETURN(
      std::unique_ptr<InteractiveSegmenter> segmenter,
      (core::VisionTaskApiFactory::Create<InteractiveSegmenter,
                                          ImageSegmenterGraphOptionsProto>(
          CreateGraphConfig(std::move(options_proto),
                            options->output_confidence_masks,
                            options->output_category_mask),
          std::move(options->base_options.op_resolver),
          core::RunningMode::IMAGE,
          /*packets_callback=*/nullptr,
          /*disable_default_service=*/
          options->base_options.disable_default_service)));
  segmenter->output_category_mask_ = options->output_category_mask;
  segmenter->output_confidence_masks_ = options->output_confidence_masks;
  return segmenter;
}

absl::StatusOr<ImageSegmenterResult> InteractiveSegmenter::Segment(
    mediapipe::Image image, const RegionOfInterest& roi,
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
  MP_ASSIGN_OR_RETURN(RenderData roi_as_render_data,
                      ConvertRoiToRenderData(roi));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, mediapipe::MakePacket<Image>(std::move(image))},
           {kRoiStreamName,
            mediapipe::MakePacket<RenderData>(std::move(roi_as_render_data))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))}}));
  std::optional<std::vector<Image>> confidence_masks;
  if (output_confidence_masks_) {
    confidence_masks =
        output_packets[kConfidenceMasksStreamName].Get<std::vector<Image>>();
  }
  std::optional<Image> category_mask;
  if (output_category_mask_) {
    category_mask = output_packets[kCategoryMaskStreamName].Get<Image>();
  }
  const std::vector<float>& quality_scores =
      output_packets[kQualityScoresStreamName].Get<std::vector<float>>();
  return {{confidence_masks, category_mask, quality_scores}};
}

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
