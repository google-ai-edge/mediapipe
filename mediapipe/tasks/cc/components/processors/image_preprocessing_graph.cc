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
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"

#include <array>
#include <complex>
#include <limits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/calculators/image/image_clone_calculator.pb.h"
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/gpu_origin.pb.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/acceleration.pb.h"
#include "mediapipe/tasks/cc/core/proto/base_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {
namespace {

using ::mediapipe::NormalizedRect;
using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::vision::ImageTensorSpecs;

constexpr char kImageTag[] = "IMAGE";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kSizeTag[] = "SIZE";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kLetterboxPaddingTag[] = "LETTERBOX_PADDING";

// Struct holding the different output streams produced by the subgraph.
struct ImagePreprocessingOutputStreams {
  Source<std::vector<Tensor>> tensors;
  Source<std::array<float, 16>> matrix;
  Source<std::array<float, 4>> letterbox_padding;
  Source<std::pair<int, int>> image_size;
  Source<Image> image;
};

// Fills in the ImageToTensorCalculatorOptions based on the ImageTensorSpecs.
absl::Status ConfigureImageToTensorCalculator(
    const ImageTensorSpecs& image_tensor_specs, GpuOrigin::Mode gpu_origin,
    mediapipe::ImageToTensorCalculatorOptions* options) {
  options->set_output_tensor_width(image_tensor_specs.image_width);
  options->set_output_tensor_height(image_tensor_specs.image_height);
  if (image_tensor_specs.tensor_type == tflite::TensorType_UINT8) {
    options->mutable_output_tensor_uint_range()->set_min(0);
    options->mutable_output_tensor_uint_range()->set_max(255);
  } else {
    const auto& normalization_options =
        image_tensor_specs.normalization_options;
    float mean = normalization_options->mean_values[0];
    float std = normalization_options->std_values[0];
    // TODO: Add support for per-channel normalization values.
    for (int i = 1; i < normalization_options->num_values; ++i) {
      if (normalization_options->mean_values[i] != mean ||
          normalization_options->std_values[i] != std) {
        return CreateStatusWithPayload(
            absl::StatusCode::kUnimplemented,
            "Per-channel image normalization is not available.");
      }
    }
    if (std::abs(std) < std::numeric_limits<float>::epsilon()) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "NormalizationOptions.std_values can't be 0. Please check if the "
          "tensor metadata has been populated correctly.");
    }
    // Deduce min and max range from normalization options by applying the
    // normalization formula to the numerical limits of uint8, i.e:
    //   output = (input - mean) / std
    options->mutable_output_tensor_float_range()->set_min((0.0f - mean) / std);
    options->mutable_output_tensor_float_range()->set_max((255.0f - mean) /
                                                          std);
  }
  // TODO: need to support different GPU origin on different
  // platforms or applications.
  options->set_gpu_origin(gpu_origin);
  return absl::OkStatus();
}

}  // namespace

bool DetermineImagePreprocessingGpuBackend(
    const core::proto::Acceleration& acceleration) {
  return acceleration.has_gpu() ||
         (acceleration.has_nnapi() &&
          acceleration.nnapi().accelerator_name() == "google-edgetpu");
}

absl::Status ConfigureImagePreprocessingGraph(
    const ModelResources& model_resources, bool use_gpu,
    proto::ImagePreprocessingGraphOptions* options) {
  return ConfigureImagePreprocessingGraph(model_resources, use_gpu,
                                          GpuOrigin::TOP_LEFT, options);
}

absl::Status ConfigureImagePreprocessingGraph(
    const ModelResources& model_resources, bool use_gpu,
    GpuOrigin::Mode gpu_origin,
    proto::ImagePreprocessingGraphOptions* options) {
  MP_ASSIGN_OR_RETURN(auto image_tensor_specs,
                      vision::BuildInputImageTensorSpecs(model_resources));
  MP_RETURN_IF_ERROR(ConfigureImageToTensorCalculator(
      image_tensor_specs, gpu_origin,
      options->mutable_image_to_tensor_options()));
  // The GPU backend isn't able to process int data. If the input tensor is
  // quantized, forces the image preprocessing graph to use CPU backend.
  if (use_gpu && image_tensor_specs.tensor_type != tflite::TensorType_UINT8) {
    options->set_backend(proto::ImagePreprocessingGraphOptions::GPU_BACKEND);
  } else {
    options->set_backend(proto::ImagePreprocessingGraphOptions::CPU_BACKEND);
  }
  return absl::OkStatus();
}

Source<Image> AddDataConverter(Source<Image> image_in, Graph& graph,
                               bool output_on_gpu) {
  auto& image_converter = graph.AddNode("ImageCloneCalculator");
  image_converter.GetOptions<mediapipe::ImageCloneCalculatorOptions>()
      .set_output_on_gpu(output_on_gpu);
  image_in >> image_converter.In("");
  return image_converter[Output<Image>("")];
}

// An ImagePreprocessingGraph performs image preprocessing.
// - Accepts CPU input images and outputs CPU tensors.
//
// Inputs:
//   IMAGE - Image
//     The image to preprocess.
//   NORM_RECT - NormalizedRect @Optional
//     Describes region of image to extract.
//     @Optional: rect covering the whole image is used if not specified.
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor populated with the converted and
//     preprocessed image.
//   MATRIX - std::array<float,16> @Optional
//     An std::array<float, 16> representing a 4x4 row-major-order matrix that
//     maps a point on the input image to a point on the output tensor, and
//     can be used to reverse the mapping by inverting the matrix.
//   LETTERBOX_PADDING - std::array<float, 4> @Optional
//     An std::array<float, 4> representing the letterbox padding from the 4
//     sides ([left, top, right, bottom]) of the output image, normalized to
//     [0.f, 1.f] by the output dimensions. The padding values are non-zero only
//     when the "keep_aspect_ratio" is true in ImagePreprocessingGraphOptions.
//   IMAGE_SIZE - std::pair<int,int> @Optional
//     The size of the original input image as a <width, height> pair.
//   IMAGE - Image @Optional
//     The image that has the pixel data stored on the target storage (CPU vs
//     GPU).
//
// The recommended way of using this subgraph is through the GraphBuilder API
// using the 'ConfigureImagePreprocessingGraph()' function. See header file for
// more details.
class ImagePreprocessingGraph : public Subgraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    Graph graph;
    auto output_streams = BuildImagePreprocessing(
        sc->Options<proto::ImagePreprocessingGraphOptions>(),
        graph[Input<Image>(kImageTag)],
        graph[Input<NormalizedRect>::Optional(kNormRectTag)], graph);
    output_streams.tensors >> graph[Output<std::vector<Tensor>>(kTensorsTag)];
    output_streams.matrix >> graph[Output<std::array<float, 16>>(kMatrixTag)];
    output_streams.letterbox_padding >>
        graph[Output<std::array<float, 4>>(kLetterboxPaddingTag)];
    output_streams.image_size >>
        graph[Output<std::pair<int, int>>(kImageSizeTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe image preprocessing subgraph into the provided
  // builder::Graph instance. The image preprocessing subgraph takes images
  // (mediapipe::Image) and region of interest (mediapipe::NormalizedRect) as
  // inputs and returns 5 output streams:
  //   - the converted tensor (mediapipe::Tensor),
  //   - the transformation matrix (std::array<float, 16>),
  //   - the letterbox padding (std::array<float, 4>>),
  //   - the original image size (std::pair<int, int>),
  //   - the image that has pixel data stored on the target storage
  //     (mediapipe::Image).
  //
  // options: the mediapipe tasks ImagePreprocessingGraphOptions.
  // image_in: (mediapipe::Image) stream to preprocess.
  // graph: the mediapipe builder::Graph instance to be updated.
  ImagePreprocessingOutputStreams BuildImagePreprocessing(
      const proto::ImagePreprocessingGraphOptions& options,
      Source<Image> image_in, Source<NormalizedRect> norm_rect_in,
      Graph& graph) {
    // Convert image to tensor.
    auto& image_to_tensor = graph.AddNode("ImageToTensorCalculator");
    image_to_tensor.GetOptions<mediapipe::ImageToTensorCalculatorOptions>()
        .CopyFrom(options.image_to_tensor_options());
    switch (options.backend()) {
      case proto::ImagePreprocessingGraphOptions::CPU_BACKEND: {
        auto cpu_image =
            AddDataConverter(image_in, graph, /*output_on_gpu=*/false);
        cpu_image >> image_to_tensor.In(kImageTag);
        break;
      }
      case proto::ImagePreprocessingGraphOptions::GPU_BACKEND: {
        auto gpu_image =
            AddDataConverter(image_in, graph, /*output_on_gpu=*/true);
        gpu_image >> image_to_tensor.In(kImageTag);
        break;
      }
      default:
        image_in >> image_to_tensor.In(kImageTag);
    }
    norm_rect_in >> image_to_tensor.In(kNormRectTag);

    // Extract optional image properties.
    auto& image_size = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_size.In(kImageTag);

    // TODO: Replace PassThroughCalculator with a calculator that
    // converts the pixel data to be stored on the target storage (CPU vs GPU).
    auto& pass_through = graph.AddNode("PassThroughCalculator");
    image_in >> pass_through.In("");

    // Connect outputs.
    return {
        /* tensors= */ image_to_tensor[Output<std::vector<Tensor>>(
            kTensorsTag)],
        /* matrix= */
        image_to_tensor[Output<std::array<float, 16>>(kMatrixTag)],
        /* letterbox_padding= */
        image_to_tensor[Output<std::array<float, 4>>(kLetterboxPaddingTag)],
        /* image_size= */ image_size[Output<std::pair<int, int>>(kSizeTag)],
        /* image= */ pass_through[Output<Image>("")],
    };
  }
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::components::processors::ImagePreprocessingGraph)

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
