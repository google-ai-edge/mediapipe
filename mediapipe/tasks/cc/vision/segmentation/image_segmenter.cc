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

#include "mediapipe/tasks/cc/vision/segmentation/image_segmenter.h"

#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/tasks/cc/core/task_api_factory.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace {

constexpr char kSegmentationStreamName[] = "segmented_mask_out";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kImageStreamName[] = "image_in";
constexpr char kImageTag[] = "IMAGE";
constexpr char kSubgraphTypeName[] =
    "mediapipe.tasks.vision.ImageSegmenterGraph";

using ::mediapipe::CalculatorGraphConfig;
using ::mediapipe::Image;

// Creates a MediaPipe graph config that only contains a single subgraph node of
// "mediapipe.tasks.vision.SegmenterGraph".
CalculatorGraphConfig CreateGraphConfig(
    std::unique_ptr<ImageSegmenterOptions> options) {
  api2::builder::Graph graph;
  auto& subgraph = graph.AddNode(kSubgraphTypeName);
  subgraph.GetOptions<ImageSegmenterOptions>().Swap(options.get());
  graph.In(kImageTag).SetName(kImageStreamName) >> subgraph.In(kImageTag);
  subgraph.Out(kGroupedSegmentationTag).SetName(kSegmentationStreamName) >>
      graph.Out(kGroupedSegmentationTag);
  return graph.GetConfig();
}

}  // namespace

absl::StatusOr<std::unique_ptr<ImageSegmenter>> ImageSegmenter::Create(
    std::unique_ptr<ImageSegmenterOptions> options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  return core::TaskApiFactory::Create<ImageSegmenter, ImageSegmenterOptions>(
      CreateGraphConfig(std::move(options)), std::move(resolver));
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
      runner_->Process({{kImageStreamName,
                         mediapipe::MakePacket<Image>(std::move(image))}}));
  return output_packets[kSegmentationStreamName].Get<std::vector<Image>>();
}

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
