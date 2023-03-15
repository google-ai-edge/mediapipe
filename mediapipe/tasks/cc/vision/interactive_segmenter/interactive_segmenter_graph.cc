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

#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/flat_color_image_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {

namespace {

using image_segmenter::proto::ImageSegmenterGraphOptions;
using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

constexpr char kSegmentationTag[] = "SEGMENTATION";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kAlphaTag[] = "ALPHA";
constexpr char kAlphaGpuTag[] = "ALPHA_GPU";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRoiTag[] = "ROI";
constexpr char kVideoTag[] = "VIDEO";

// Updates the graph to return `roi` stream which has same dimension as
// `image`, and rendered with `roi`. If `use_gpu` is true, returned `Source` is
// in GpuBuffer format, otherwise using ImageFrame.
Source<> RoiToAlpha(Source<Image> image, Source<RenderData> roi, bool use_gpu,
                    Graph& graph) {
  // TODO: Replace with efficient implementation.
  const absl::string_view image_tag_with_suffix =
      use_gpu ? kImageGpuTag : kImageCpuTag;

  // Generates a blank canvas with same size as input image.
  auto& flat_color = graph.AddNode("FlatColorImageCalculator");
  auto& flat_color_options =
      flat_color.GetOptions<FlatColorImageCalculatorOptions>();
  // SetAlphaCalculator only takes 1st channel.
  flat_color_options.mutable_color()->set_r(0);
  image >> flat_color.In(kImageTag)[0];
  auto blank_canvas = flat_color.Out(kImageTag)[0];

  auto& from_mp_image = graph.AddNode("FromImageCalculator");
  blank_canvas >> from_mp_image.In(kImageTag);
  auto blank_canvas_in_cpu_or_gpu = from_mp_image.Out(image_tag_with_suffix);

  auto& roi_to_alpha = graph.AddNode("AnnotationOverlayCalculator");
  blank_canvas_in_cpu_or_gpu >>
      roi_to_alpha.In(use_gpu ? kImageGpuTag : kImageTag);
  roi >> roi_to_alpha.In(0);
  auto alpha = roi_to_alpha.Out(use_gpu ? kImageGpuTag : kImageTag);

  return alpha;
}

}  // namespace

// An "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph"
// performs semantic segmentation given user's region-of-interest. Two kinds of
// outputs are provided: SEGMENTATION and GROUPED_SEGMENTATION. Users can
// retrieve segmented mask of only particular category/channel from
// SEGMENTATION, and users can also get all segmented masks from
// GROUPED_SEGMENTATION.
// - Accepts CPU input images and outputs segmented masks on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform segmentation on.
//   ROI - RenderData proto
//     Region of interest based on user interaction. Currently only support
//     Point format, and Color has to be (255, 255, 255).
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection
//     on.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   SEGMENTATION - mediapipe::Image @Multiple
//     Segmented masks for individual category. Segmented mask of single
//     category can be accessed by index based output stream.
//   GROUPED_SEGMENTATION - std::vector<mediapipe::Image>
//     The output segmented masks grouped in a vector.
//   IMAGE - mediapipe::Image
//     The image that image segmenter runs on.
//
// Example:
// node {
//   calculator:
//   "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph"
//   input_stream: "IMAGE:image"
//   input_stream: "ROI:region_of_interest"
//   output_stream: "SEGMENTATION:segmented_masks"
//   options {
//     [mediapipe.tasks.vision.image_segmenter.proto.ImageSegmenterGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "/path/to/model.tflite"
//         }
//       }
//       segmenter_options {
//         output_type: CONFIDENCE_MASK
//       }
//     }
//   }
// }
class InteractiveSegmenterGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    const auto& task_options = sc->Options<ImageSegmenterGraphOptions>();
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());

    Source<Image> image = graph[Input<Image>(kImageTag)];
    Source<RenderData> roi = graph[Input<RenderData>(kRoiTag)];
    Source<NormalizedRect> norm_rect =
        graph[Input<NormalizedRect>(kNormRectTag)];
    const absl::string_view image_tag_with_suffix =
        use_gpu ? kImageGpuTag : kImageCpuTag;
    const absl::string_view alpha_tag_with_suffix =
        use_gpu ? kAlphaGpuTag : kAlphaTag;

    auto& from_mp_image = graph.AddNode("FromImageCalculator");
    image >> from_mp_image.In(kImageTag);
    auto image_in_cpu_or_gpu = from_mp_image.Out(image_tag_with_suffix);

    auto alpha_in_cpu_or_gpu = RoiToAlpha(image, roi, use_gpu, graph);

    auto& set_alpha = graph.AddNode("SetAlphaCalculator");
    image_in_cpu_or_gpu >> set_alpha.In(use_gpu ? kImageGpuTag : kImageTag);
    alpha_in_cpu_or_gpu >> set_alpha.In(alpha_tag_with_suffix);
    auto image_in_cpu_or_gpu_with_set_alpha =
        set_alpha.Out(use_gpu ? kImageGpuTag : kImageTag);

    auto& to_mp_image = graph.AddNode("ToImageCalculator");
    image_in_cpu_or_gpu_with_set_alpha >> to_mp_image.In(image_tag_with_suffix);
    auto image_with_set_alpha = to_mp_image.Out(kImageTag);

    auto& image_segmenter = graph.AddNode(
        "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph");
    image_segmenter.GetOptions<ImageSegmenterGraphOptions>() = task_options;
    image_with_set_alpha >> image_segmenter.In(kImageTag);
    norm_rect >> image_segmenter.In(kNormRectTag);

    image_segmenter.Out(kSegmentationTag) >>
        graph[Output<Image>(kSegmentationTag)];
    image_segmenter.Out(kGroupedSegmentationTag) >>
        graph[Output<std::vector<Image>>(kGroupedSegmentationTag)];
    image_segmenter.Out(kImageTag) >> graph[Output<Image>(kImageTag)];

    return graph.GetConfig();
  }
};

// REGISTER_MEDIAPIPE_GRAPH argument has to fit on one line to work properly.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::interactive_segmenter::InteractiveSegmenterGraph);
// clang-format on

}  // namespace interactive_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
