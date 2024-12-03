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

#include "mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.h"

#include <optional>
#include <utility>

#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/vision/core/image_processing_options.h"
#include "mediapipe/tasks/cc/vision/core/running_mode.h"
#include "mediapipe/tasks/cc/vision/core/vision_task_api_factory.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/util/label_map.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {
namespace {

constexpr char kConfidenceMasksTag[] = "CONFIDENCE_MASKS";
constexpr char kConfidenceMasksStreamName[] = "confidence_masks";
constexpr char kCategoryMaskTag[] = "CATEGORY_MASK";
constexpr char kCategoryMaskStreamName[] = "category_mask";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kOutputSizeStreamName[] = "output_size";
constexpr char kImageInStreamName[] = "image_in";
constexpr char kImageOutStreamName[] = "image_out";
constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectStreamName[] = "norm_rect_in";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kQualityScoresStreamName[] = "quality_scores";
constexpr char kQualityScoresTag[] = "QUALITY_SCORES";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph";
constexpr int kMicroSecondsPerMilliSecond = 1000;

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ImageSegmenterGraphOptionsProto = ::mediapipe::tasks::vision::
    image_segmenter::proto::ImageSegmenterGraphOptions;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageSegmenterGraphOptionsProto> options,
    bool output_confidence_masks, bool output_category_mask,
    bool enable_flow_limiting) {
  api2::builder::Graph graph;
  auto& task_subgraph = graph.AddNode(kSubgraphTypeName);
  task_subgraph.GetOptions<ImageSegmenterGraphOptionsProto>().Swap(
      options.get());
  graph.In(kImageTag).SetName(kImageInStreamName);
  graph.In(kNormRectTag).SetName(kNormRectStreamName);
  graph.In(kOutputSizeTag).SetName(kOutputSizeStreamName);
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
  if (enable_flow_limiting) {
    return tasks::core::AddFlowLimiterCalculator(
        graph, task_subgraph, {kImageTag, kNormRectTag, kOutputSizeTag},
        kConfidenceMasksTag);
  }
  graph.In(kImageTag) >> task_subgraph.In(kImageTag);
  graph.In(kNormRectTag) >> task_subgraph.In(kNormRectTag);
  graph.In(kOutputSizeTag) >> task_subgraph.In(kOutputSizeTag);
  return graph.GetConfig();
}

// Converts the user-facing ImageSegmenterOptions struct to the internal
// ImageSegmenterOptions proto.
std::unique_ptr<ImageSegmenterGraphOptionsProto>
ConvertImageSegmenterOptionsToProto(ImageSegmenterOptions* options) {
  auto options_proto = std::make_unique<ImageSegmenterGraphOptionsProto>();
  auto base_options_proto = std::make_unique<tasks::core::proto::BaseOptions>(
      tasks::core::ConvertBaseOptionsToProto(&(options->base_options)));
  options_proto->mutable_base_options()->Swap(base_options_proto.get());
  options_proto->mutable_base_options()->set_use_stream_mode(
      options->running_mode != core::RunningMode::IMAGE);
  options_proto->set_display_names_locale(options->display_names_locale);
  return options_proto;
}

absl::StatusOr<std::vector<std::string>> GetLabelsFromGraphConfig(
    const CalculatorGraphConfig& graph_config) {
  bool found_tensor_to_segmentation_calculator = false;
  std::vector<std::string> labels;
  for (const auto& node : graph_config.node()) {
    if (node.calculator() ==
        "mediapipe.tasks.TensorsToSegmentationCalculator") {
      if (!found_tensor_to_segmentation_calculator) {
        found_tensor_to_segmentation_calculator = true;
      } else {
        return absl::Status(CreateStatusWithPayload(
            absl::StatusCode::kFailedPrecondition,
            "The graph has more than one "
            "mediapipe.tasks.TensorsToSegmentationCalculator."));
      }
      TensorsToSegmentationCalculatorOptions options =
          node.options().GetExtension(
              TensorsToSegmentationCalculatorOptions::ext);
      if (!options.label_items().empty()) {
        for (int i = 0; i < options.label_items_size(); ++i) {
          if (!options.label_items().contains(i)) {
            return absl::Status(CreateStatusWithPayload(
                absl::StatusCode::kFailedPrecondition,
                absl::StrFormat("The lablemap have no expected key: %d.", i)));
          }
          labels.push_back(options.label_items().at(i).name());
        }
      }
    }
  }
  return labels;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageSegmenter>> ImageSegmenter::Create(
    std::unique_ptr<ImageSegmenterOptions> options) {
  if (!options->output_confidence_masks && !options->output_category_mask) {
    return absl::InvalidArgumentError(
        "At least one of `output_confidence_masks` and `output_category_mask` "
        "must be set.");
  }
  auto options_proto = ConvertImageSegmenterOptionsToProto(options.get());
  tasks::core::PacketsCallback packets_callback = nullptr;
  if (options->result_callback) {
    auto result_callback = options->result_callback;
    bool output_category_mask = options->output_category_mask;
    bool output_confidence_masks = options->output_confidence_masks;
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
          std::optional<std::vector<Image>> confidence_masks;
          if (output_confidence_masks) {
            confidence_masks =
                status_or_packets.value()[kConfidenceMasksStreamName]
                    .Get<std::vector<Image>>();
          }
          std::optional<Image> category_mask;
          if (output_category_mask) {
            category_mask =
                status_or_packets.value()[kCategoryMaskStreamName].Get<Image>();
          }
          const std::vector<float>& quality_scores =
              status_or_packets.value()[kQualityScoresStreamName]
                  .Get<std::vector<float>>();
          Packet image_packet = status_or_packets.value()[kImageOutStreamName];
          result_callback(
              {{confidence_masks, category_mask, quality_scores}},
              image_packet.Get<Image>(),
              image_packet.Timestamp().Value() / kMicroSecondsPerMilliSecond);
        };
  }
  auto image_segmenter =
      core::VisionTaskApiFactory::Create<ImageSegmenter,
                                         ImageSegmenterGraphOptionsProto>(
          CreateGraphConfig(
              std::move(options_proto), options->output_confidence_masks,
              options->output_category_mask,
              options->running_mode == core::RunningMode::LIVE_STREAM),
          std::move(options->base_options.op_resolver), options->running_mode,
          std::move(packets_callback),
          /*disable_default_service=*/
          options->base_options.disable_default_service);
  if (!image_segmenter.ok()) {
    return image_segmenter.status();
  }
  image_segmenter.value()->output_confidence_masks_ =
      options->output_confidence_masks;
  image_segmenter.value()->output_category_mask_ =
      options->output_category_mask;
  MP_ASSIGN_OR_RETURN(
      (*image_segmenter)->labels_,
      GetLabelsFromGraphConfig((*image_segmenter)->runner_->GetGraphConfig()));
  return image_segmenter;
}

absl::StatusOr<ImageSegmenterResult> ImageSegmenter::Segment(
    mediapipe::Image image,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  return Segment(image, {
                            /*output_width=*/image.width(),
                            /*output_height=*/image.height(),
                            std::move(image_processing_options),
                        });
}

absl::StatusOr<ImageSegmenterResult> ImageSegmenter::Segment(
    mediapipe::Image image, SegmentationOptions segmentation_options) {
  MP_RETURN_IF_ERROR(ValidateSegmentationOptions(segmentation_options));
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(
                          segmentation_options.image_processing_options, image,
                          /*roi_allowed=*/false));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessImageData(
          {{kImageInStreamName, mediapipe::MakePacket<Image>(std::move(image))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))},
           {kOutputSizeStreamName,
            MakePacket<std::pair<int, int>>(
                std::make_pair(segmentation_options.output_width,
                               segmentation_options.output_height))}}));
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

absl::StatusOr<ImageSegmenterResult> ImageSegmenter::SegmentForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  return SegmentForVideo(image, timestamp_ms,
                         {
                             /*output_width=*/image.width(),
                             /*output_height=*/image.height(),
                             std::move(image_processing_options),
                         });
}

absl::StatusOr<ImageSegmenterResult> ImageSegmenter::SegmentForVideo(
    mediapipe::Image image, int64_t timestamp_ms,
    SegmentationOptions segmentation_options) {
  MP_RETURN_IF_ERROR(ValidateSegmentationOptions(segmentation_options));
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(
                          segmentation_options.image_processing_options, image,
                          /*roi_allowed=*/false));
  MP_ASSIGN_OR_RETURN(
      auto output_packets,
      ProcessVideoData(
          {{kImageInStreamName,
            MakePacket<Image>(std::move(image))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
           {kNormRectStreamName,
            MakePacket<NormalizedRect>(std::move(norm_rect))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
           {kOutputSizeStreamName,
            MakePacket<std::pair<int, int>>(
                std::make_pair(segmentation_options.output_width,
                               segmentation_options.output_height))
                .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}}));
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

absl::Status ImageSegmenter::SegmentAsync(
    Image image, int64_t timestamp_ms,
    std::optional<core::ImageProcessingOptions> image_processing_options) {
  return SegmentAsync(image, timestamp_ms,
                      {
                          /*output_width=*/image.width(),
                          /*output_height=*/image.height(),
                          std::move(image_processing_options),
                      });
}

absl::Status ImageSegmenter::SegmentAsync(
    Image image, int64_t timestamp_ms,
    SegmentationOptions segmentation_options) {
  MP_RETURN_IF_ERROR(ValidateSegmentationOptions(segmentation_options));
  if (image.UsesGpu()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrCat("GPU input images are currently not supported."),
        MediaPipeTasksStatus::kRunnerUnexpectedInputError);
  }
  MP_ASSIGN_OR_RETURN(NormalizedRect norm_rect,
                      ConvertToNormalizedRect(
                          segmentation_options.image_processing_options, image,
                          /*roi_allowed=*/false));
  return SendLiveStreamData(
      {{kImageInStreamName,
        MakePacket<Image>(std::move(image))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kNormRectStreamName,
        MakePacket<NormalizedRect>(std::move(norm_rect))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))},
       {kOutputSizeStreamName,
        MakePacket<std::pair<int, int>>(
            std::make_pair(segmentation_options.output_width,
                           segmentation_options.output_height))
            .At(Timestamp(timestamp_ms * kMicroSecondsPerMilliSecond))}});
}

}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
