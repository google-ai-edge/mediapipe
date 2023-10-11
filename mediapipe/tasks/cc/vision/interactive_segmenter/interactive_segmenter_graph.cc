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

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/util/flat_color_image_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/graph_builder_utils.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace interactive_segmenter {
namespace internal {

// A calculator to add thickness to the render data according to the image size,
// so that the render data is scale invariant to the image size. If the render
// data already has thickness, it will be kept as is.
class AddThicknessToRenderDataCalculator : public api2::Node {
 public:
  static constexpr api2::Input<Image> kImageIn{"IMAGE"};
  static constexpr api2::Input<mediapipe::RenderData> kRenderDataIn{
      "RENDER_DATA"};
  static constexpr api2::Output<mediapipe::RenderData> kRenderDataOut{
      "RENDER_DATA"};

  static constexpr int kModelInputTensorWidth = 512;
  static constexpr int kModelInputTensorHeight = 512;

  MEDIAPIPE_NODE_CONTRACT(kImageIn, kRenderDataIn, kRenderDataOut);

  absl::Status Process(CalculatorContext* cc) final {
    mediapipe::RenderData render_data = kRenderDataIn(cc).Get();
    Image image = kImageIn(cc).Get();
    double thickness = std::max(
        std::max(image.width() / static_cast<double>(kModelInputTensorWidth),
                 image.height() / static_cast<double>(kModelInputTensorHeight)),
        1.0);

    for (auto& annotation : *render_data.mutable_render_annotations()) {
      if (!annotation.has_thickness()) {
        annotation.set_thickness(thickness);
      }
    }
    kRenderDataOut(cc).Send(render_data);
    return absl::OkStatus();
  }
};

// NOLINTBEGIN: Node registration doesn't work when part of calculator name is
// moved to next line.
// clang-format off
MEDIAPIPE_REGISTER_NODE(
    ::mediapipe::tasks::vision::interactive_segmenter::internal::AddThicknessToRenderDataCalculator);
// clang-format on
// NOLINTEND

}  // namespace internal

namespace {

using image_segmenter::proto::ImageSegmenterGraphOptions;
using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;

constexpr absl::string_view kSegmentationTag{"SEGMENTATION"};
constexpr absl::string_view kGroupedSegmentationTag{"GROUPED_SEGMENTATION"};
constexpr absl::string_view kConfidenceMaskTag{"CONFIDENCE_MASK"};
constexpr absl::string_view kConfidenceMasksTag{"CONFIDENCE_MASKS"};
constexpr absl::string_view kCategoryMaskTag{"CATEGORY_MASK"};
constexpr absl::string_view kImageTag{"IMAGE"};
constexpr absl::string_view kImageCpuTag{"IMAGE_CPU"};
constexpr absl::string_view kImageGpuTag{"IMAGE_GPU"};
constexpr absl::string_view kAlphaTag{"ALPHA"};
constexpr absl::string_view kAlphaGpuTag{"ALPHA_GPU"};
constexpr absl::string_view kNormRectTag{"NORM_RECT"};
constexpr absl::string_view kRoiTag{"ROI"};
constexpr absl::string_view kQualityScoresTag{"QUALITY_SCORES"};
constexpr absl::string_view kRenderDataTag{"RENDER_DATA"};

// Updates the graph to return `roi` stream which has same dimension as
// `image`, and rendered with `roi`. If `use_gpu` is true, returned `Source` is
// in GpuBuffer format, otherwise using ImageFrame.
Source<> RoiToAlpha(Source<Image> image, Source<RenderData> roi, bool use_gpu,
                    Graph& graph) {
  // TODO: Replace with efficient implementation.
  const absl::string_view image_tag_with_suffix =
      use_gpu ? kImageGpuTag : kImageCpuTag;

  // Adds thickness to the render data so that the render data is scale
  // invariant to the input image size.
  auto& add_thickness = graph.AddNode(
      "mediapipe::tasks::vision::interactive_segmenter::internal::"
      "AddThicknessToRenderDataCalculator");
  image >> add_thickness.In(kImageTag);
  roi >> add_thickness.In(kRenderDataTag);
  auto roi_with_thickness = add_thickness.Out(kRenderDataTag);

  // Generates a blank canvas with same size as input image.
  auto& flat_color = graph.AddNode("FlatColorImageCalculator");
  auto& flat_color_options =
      flat_color.GetOptions<FlatColorImageCalculatorOptions>();
  // SetAlphaCalculator only takes 1st channel.
  flat_color_options.mutable_color()->set_r(0);
  image >> flat_color.In(kImageTag);
  auto blank_canvas = flat_color.Out(kImageTag);

  auto& from_mp_image = graph.AddNode("FromImageCalculator");
  blank_canvas >> from_mp_image.In(kImageTag);
  auto blank_canvas_in_cpu_or_gpu = from_mp_image.Out(image_tag_with_suffix);

  auto& roi_to_alpha = graph.AddNode("AnnotationOverlayCalculator");
  blank_canvas_in_cpu_or_gpu >>
      roi_to_alpha.In(use_gpu ? kImageGpuTag : kImageTag);
  roi_with_thickness >> roi_to_alpha.In(0);
  auto alpha = roi_to_alpha.Out(use_gpu ? kImageGpuTag : kImageTag);

  return alpha;
}

}  // namespace

// An "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph"
// performs semantic segmentation given the user's region-of-interest. The graph
// can output optional confidence masks if CONFIDENCE_MASKS is connected, and an
// optional category mask if CATEGORY_MASK is connected. At least one of
// CONFIDENCE_MASK, CONFIDENCE_MASKS and CATEGORY_MASK must be connected.
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
//   CONFIDENCE_MASK - mediapipe::Image @Multiple
//     Confidence masks for individual category. Confidence mask of single
//     category can be accessed by index based output stream.
//   CONFIDENCE_MASKS - std::vector<mediapipe::Image> @Optional
//     The output confidence masks grouped in a vector.
//   CATEGORY_MASK - mediapipe::Image @Optional
//     Optional Category mask.
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

    // Creates an RGBA image with model input tensor size.
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

    // TODO: remove deprecated output type support.
    if (task_options.segmenter_options().has_output_type()) {
      image_segmenter.Out(kSegmentationTag) >>
          graph[Output<Image>(kSegmentationTag)];
      image_segmenter.Out(kGroupedSegmentationTag) >>
          graph[Output<std::vector<Image>>(kGroupedSegmentationTag)];
    } else {
      if (HasOutput(sc->OriginalNode(), kConfidenceMaskTag)) {
        image_segmenter.Out(kConfidenceMaskTag) >>
            graph[Output<Image>(kConfidenceMaskTag)];
      }
      if (HasOutput(sc->OriginalNode(), kConfidenceMasksTag)) {
        image_segmenter.Out(kConfidenceMasksTag) >>
            graph[Output<Image>(kConfidenceMasksTag)];
      }
      if (HasOutput(sc->OriginalNode(), kCategoryMaskTag)) {
        image_segmenter.Out(kCategoryMaskTag) >>
            graph[Output<Image>(kCategoryMaskTag)];
      }
    }
    image_segmenter.Out(kQualityScoresTag) >>
        graph[Output<std::vector<float>>::Optional(kQualityScoresTag)];
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
