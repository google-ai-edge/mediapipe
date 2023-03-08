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

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/calculators/image/image_cropping_calculator.pb.h"
#include "mediapipe/calculators/image/warp_affine_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/vision/face_stylizer/calculators/tensors_to_image_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/face_stylizer/proto/face_stylizer_graph_options.pb.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace face_stylizer {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::TensorsToImageCalculatorOptions;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::vision::face_stylizer::proto::
    FaceStylizerGraphOptions;

constexpr char kImageTag[] = "IMAGE";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kStylizedImageTag[] = "STYLIZED_IMAGE";
constexpr char kTensorsTag[] = "TENSORS";

// Struct holding the different output streams produced by the face stylizer
// graph.
struct FaceStylizerOutputStreams {
  Source<Image> stylized_image;
  Source<Image> original_image;
};

void ConfigureTensorsToImageCalculator(
    const ImageToTensorCalculatorOptions& image_to_tensor_options,
    TensorsToImageCalculatorOptions* tensors_to_image_options) {
  tensors_to_image_options->set_gpu_origin(mediapipe::GpuOrigin_Mode_TOP_LEFT);
  if (image_to_tensor_options.has_output_tensor_float_range()) {
    auto* mutable_range =
        tensors_to_image_options->mutable_input_tensor_float_range();
    // TODO: Make the float range flexiable.
    mutable_range->set_min(0);
    mutable_range->set_max(1);
  } else if (image_to_tensor_options.has_output_tensor_uint_range()) {
    auto* mutable_range =
        tensors_to_image_options->mutable_input_tensor_uint_range();
    const auto& reference_range =
        image_to_tensor_options.output_tensor_uint_range();
    mutable_range->set_min(reference_range.min());
    mutable_range->set_max(reference_range.max());
  }
}

}  // namespace

// A "mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph" performs face
// stylization.
//
// Inputs:
//   IMAGE - Image
//     Image to perform face stylization on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to perform classification on.
//     @Optional: rect covering the whole image is used if not specified.
//
// Outputs:
//   IMAGE - mediapipe::Image
//     The face stylization output image.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph"
//   input_stream: "IMAGE:image_in"
//   input_stream: "NORM_RECT:norm_rect"
//   output_stream: "IMAGE:image_out"
//   output_stream: "STYLIZED_IMAGE:stylized_image"
//   options {
//     [mediapipe.tasks.vision.face_stylizer.proto.FaceStylizerGraphOptions.ext]
//     {
//       base_options {
//         model_asset {
//           file_name: "face_stylization.tflite"
//         }
//       }
//     }
//   }
// }
class FaceStylizerGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<FaceStylizerGraphOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(
        auto output_streams,
        BuildFaceStylizerGraph(
            sc->Options<FaceStylizerGraphOptions>(), *model_resources,
            graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph));
    output_streams.stylized_image >> graph[Output<Image>(kStylizedImageTag)];
    output_streams.original_image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<FaceStylizerOutputStreams> BuildFaceStylizerGraph(
      const FaceStylizerGraphOptions& task_options,
      const ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in, Graph& graph) {
    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu,
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    auto& image_to_tensor_options =
        *preprocessing
             .GetOptions<components::processors::proto::
                             ImagePreprocessingGraphOptions>()
             .mutable_image_to_tensor_options();
    image_to_tensor_options.set_keep_aspect_ratio(true);
    image_to_tensor_options.set_border_mode(
        mediapipe::ImageToTensorCalculatorOptions::BORDER_ZERO);
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);
    auto preprocessed_tensors = preprocessing.Out(kTensorsTag);
    auto transform_matrix = preprocessing.Out(kMatrixTag);
    auto image_size = preprocessing.Out(kImageSizeTag);

    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    preprocessed_tensors >> inference.In(kTensorsTag);
    auto model_output_tensors =
        inference.Out(kTensorsTag).Cast<std::vector<Tensor>>();

    auto& tensors_to_image =
        graph.AddNode("mediapipe.tasks.TensorsToImageCalculator");
    ConfigureTensorsToImageCalculator(
        image_to_tensor_options,
        &tensors_to_image.GetOptions<TensorsToImageCalculatorOptions>());
    model_output_tensors >> tensors_to_image.In(kTensorsTag);
    auto tensor_image = tensors_to_image.Out(kImageTag);

    auto& inverse_matrix = graph.AddNode("InverseMatrixCalculator");
    transform_matrix >> inverse_matrix.In(kMatrixTag);
    auto inverse_transform_matrix = inverse_matrix.Out(kMatrixTag);

    auto& warp_affine = graph.AddNode("WarpAffineCalculator");
    auto& warp_affine_options =
        warp_affine.GetOptions<WarpAffineCalculatorOptions>();
    warp_affine_options.set_border_mode(
        WarpAffineCalculatorOptions::BORDER_ZERO);
    warp_affine_options.set_gpu_origin(mediapipe::GpuOrigin_Mode_TOP_LEFT);
    tensor_image >> warp_affine.In(kImageTag);
    inverse_transform_matrix >> warp_affine.In(kMatrixTag);
    image_size >> warp_affine.In(kOutputSizeTag);
    auto image_to_crop = warp_affine.Out(kImageTag);

    // The following calculators are for cropping and resizing the output image
    // based on the roi and the model output size. As the WarpAffineCalculator
    // rotates the image based on the transform matrix, the rotation info in the
    // rect proto is stripped to prevent the ImageCroppingCalculator from
    // performing extra rotation.
    auto& strip_rotation =
        graph.AddNode("mediapipe.tasks.StripRotationCalculator");
    norm_rect_in >> strip_rotation.In(kNormRectTag);
    auto norm_rect_no_rotation = strip_rotation.Out(kNormRectTag);
    auto& from_image = graph.AddNode("FromImageCalculator");
    image_to_crop >> from_image.In(kImageTag);
    auto& image_cropping = graph.AddNode("ImageCroppingCalculator");
    auto& image_cropping_opts =
        image_cropping.GetOptions<ImageCroppingCalculatorOptions>();
    image_cropping_opts.set_output_max_width(
        image_to_tensor_options.output_tensor_width());
    image_cropping_opts.set_output_max_height(
        image_to_tensor_options.output_tensor_height());
    norm_rect_no_rotation >> image_cropping.In(kNormRectTag);
    auto& to_image = graph.AddNode("ToImageCalculator");
    // ImageCroppingCalculator currently doesn't support mediapipe::Image, the
    // graph selects its cpu or gpu path based on the image preprocessing
    // backend.
    if (use_gpu) {
      from_image.Out(kImageGpuTag) >> image_cropping.In(kImageGpuTag);
      image_cropping.Out(kImageGpuTag) >> to_image.In(kImageGpuTag);
    } else {
      from_image.Out(kImageCpuTag) >> image_cropping.In(kImageTag);
      image_cropping.Out(kImageTag) >> to_image.In(kImageCpuTag);
    }

    return {{/*stylized_image=*/to_image.Out(kImageTag).Cast<Image>(),
             /*original_image=*/preprocessing.Out(kImageTag).Cast<Image>()}};
  }
};

// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::vision::face_stylizer::FaceStylizerGraph);  // NOLINT
// clang-format on

}  // namespace face_stylizer
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
