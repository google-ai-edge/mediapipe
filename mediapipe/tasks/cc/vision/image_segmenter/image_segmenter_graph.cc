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

#include <memory>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/tensor/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/image_preprocessing.h"
#include "mediapipe/tasks/cc/components/image_preprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::Image;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::MultiSource;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::SegmenterOptions;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::mediapipe::tasks::vision::image_segmenter::proto::ImageSegmenterOptions;
using ::tflite::Tensor;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64, ::mediapipe::LabelMapItem>;

constexpr char kSegmentationTag[] = "SEGMENTATION";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kImageTag[] = "IMAGE";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";

// Struct holding the different output streams produced by the image segmenter
// subgraph.
struct ImageSegmenterOutputs {
  std::vector<Source<Image>> segmented_masks;
  // The same as the input image, mainly used for live stream mode.
  Source<Image> image;
};

}  // namespace

absl::Status SanityCheckOptions(const ImageSegmenterOptions& options) {
  if (options.segmenter_options().output_type() ==
      SegmenterOptions::UNSPECIFIED) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "`output_type` must not be UNSPECIFIED",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::StatusOr<LabelItems> GetLabelItemsIfAny(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata, absl::string_view locale) {
  const std::string labels_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_AXIS_LABELS);
  if (labels_filename.empty()) {
    LabelItems empty_label_items;
    return empty_label_items;
  }
  ASSIGN_OR_RETURN(absl::string_view labels_file,
                   metadata_extractor.GetAssociatedFile(labels_filename));
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_AXIS_LABELS,
          locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    ASSIGN_OR_RETURN(display_names_file, metadata_extractor.GetAssociatedFile(
                                             display_names_filename));
  }
  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::Status ConfigureTensorsToSegmentationCalculator(
    const ImageSegmenterOptions& segmenter_option,
    const core::ModelResources& model_resources,
    TensorsToSegmentationCalculatorOptions* options) {
  *options->mutable_segmenter_options() = segmenter_option.segmenter_options();
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Segmentation tflite models are assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  if (primary_subgraph->outputs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Segmentation tflite models are assumed to have a single output.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }

  const ModelMetadataExtractor* metadata_extractor =
      model_resources.GetMetadataExtractor();
  ASSIGN_OR_RETURN(
      *options->mutable_label_items(),
      GetLabelItemsIfAny(*metadata_extractor,
                         *metadata_extractor->GetOutputTensorMetadata()->Get(0),
                         segmenter_option.display_names_locale()));
  return absl::OkStatus();
}

absl::StatusOr<const Tensor*> GetOutputTensor(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  const auto* output_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->outputs())[0]];
  return output_tensor;
}

// An "mediapipe.tasks.vision.ImageSegmenterGraph" performs semantic
// segmentation.
// Two kinds of outputs are provided: SEGMENTATION and GROUPED_SEGMENTATION.
// Users can retrieve segmented mask of only particular category/channel from
// SEGMENTATION, and users can also get all segmented masks from
// GROUPED_SEGMENTATION.
// - Accepts CPU input images and outputs segmented masks on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform segmentation on.
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
//   calculator: "mediapipe.tasks.vision.ImageSegmenterGraph"
//   input_stream: "IMAGE:image"
//   output_stream: "SEGMENTATION:segmented_masks"
//   options {
//     [mediapipe.tasks.vision.image_segmenter.proto.ImageSegmenterOptions.ext]
//     {
//       segmenter_options {
//         output_type: CONFIDENCE_MASK
//         activation: SOFTMAX
//       }
//     }
//   }
// }
class ImageSegmenterGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<ImageSegmenterOptions>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(auto output_streams,
                     BuildSegmentationTask(
                         sc->Options<ImageSegmenterOptions>(), *model_resources,
                         graph[Input<Image>(kImageTag)], graph));

    auto& merge_images_to_vector =
        graph.AddNode("MergeImagesToVectorCalculator");
    for (int i = 0; i < output_streams.segmented_masks.size(); ++i) {
      output_streams.segmented_masks[i] >>
          merge_images_to_vector[Input<Image>::Multiple("")][i];
      output_streams.segmented_masks[i] >>
          graph[Output<Image>::Multiple(kSegmentationTag)][i];
    }
    merge_images_to_vector.Out("") >>
        graph[Output<std::vector<Image>>(kGroupedSegmentationTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe image segmentation task pipeline graph into the provided
  // builder::Graph instance. The segmentation pipeline takes images
  // (mediapipe::Image) as the input and returns segmented image mask as output.
  //
  // task_options: the mediapipe tasks ImageSegmenterOptions proto.
  // model_resources: the ModelSources object initialized from a segmentation
  // model file with model metadata.
  // image_in: (mediapipe::Image) stream to run segmentation on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ImageSegmenterOutputs> BuildSegmentationTask(
      const ImageSegmenterOptions& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));

    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    auto& preprocessing =
        graph.AddNode("mediapipe.tasks.ImagePreprocessingSubgraph");
    MP_RETURN_IF_ERROR(ConfigureImagePreprocessing(
        model_resources,
        &preprocessing.GetOptions<ImagePreprocessingOptions>()));
    image_in >> preprocessing.In(kImageTag);

    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(model_resources, graph);
    preprocessing.Out(kTensorsTag) >> inference.In(kTensorsTag);

    // Adds segmentation calculators for output streams.
    auto& tensor_to_images = graph.AddNode("TensorsToSegmentationCalculator");
    RET_CHECK_OK(ConfigureTensorsToSegmentationCalculator(
        task_options, model_resources,
        &tensor_to_images
             .GetOptions<TensorsToSegmentationCalculatorOptions>()));
    inference.Out(kTensorsTag) >> tensor_to_images.In(kTensorsTag);

    // Adds image property calculator for output size.
    auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
    image_in >> image_properties.In("IMAGE");
    image_properties.Out("SIZE") >> tensor_to_images.In(kOutputSizeTag);

    // Exports multiple segmented masks.
    std::vector<Source<Image>> segmented_masks;
    if (task_options.segmenter_options().output_type() ==
        SegmenterOptions::CATEGORY_MASK) {
      segmented_masks.push_back(
          Source<Image>(tensor_to_images[Output<Image>(kSegmentationTag)]));
    } else {
      ASSIGN_OR_RETURN(const Tensor* output_tensor,
                       GetOutputTensor(model_resources));
      const int segmentation_streams_num = *output_tensor->shape()->rbegin();
      for (int i = 0; i < segmentation_streams_num; ++i) {
        segmented_masks.push_back(Source<Image>(
            tensor_to_images[Output<Image>::Multiple(kSegmentationTag)][i]));
      }
    }
    return {{
        .segmented_masks = segmented_masks,
        .image = preprocessing[Output<Image>(kImageTag)],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::ImageSegmenterGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
