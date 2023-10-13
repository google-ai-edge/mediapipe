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

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/calculators/image/image_clone_calculator.pb.h"
#include "mediapipe/calculators/image/image_transformation_calculator.pb.h"
#include "mediapipe/calculators/image/set_alpha_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_converter_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/image_preprocessing_graph.h"
#include "mediapipe/tasks/cc/components/processors/proto/image_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/calculators/tensors_to_segmentation_calculator.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/image_segmenter_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/image_segmenter/proto/segmenter_options.pb.h"
#include "mediapipe/tasks/cc/vision/utils/image_tensor_specs.h"
#include "mediapipe/tasks/metadata/image_segmenter_metadata_schema_generated.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/graph_builder_utils.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace vision {
namespace image_segmenter {

namespace {

using ::mediapipe::Image;
using ::mediapipe::NormalizedRect;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::MultiSource;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::mediapipe::tasks::vision::image_segmenter::proto::
    ImageSegmenterGraphOptions;
using ::mediapipe::tasks::vision::image_segmenter::proto::SegmenterOptions;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64_t, ::mediapipe::LabelMapItem>;

constexpr char kSegmentationTag[] = "SEGMENTATION";
constexpr char kGroupedSegmentationTag[] = "GROUPED_SEGMENTATION";
constexpr char kConfidenceMaskTag[] = "CONFIDENCE_MASK";
constexpr char kConfidenceMasksTag[] = "CONFIDENCE_MASKS";
constexpr char kCategoryMaskTag[] = "CATEGORY_MASK";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageCpuTag[] = "IMAGE_CPU";
constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kOutputSizeTag[] = "OUTPUT_SIZE";
constexpr char kSizeTag[] = "SIZE";
constexpr char kQualityScoresTag[] = "QUALITY_SCORES";
constexpr char kSegmentationMetadataName[] = "SEGMENTER_METADATA";

// Struct holding the different output streams produced by the image segmenter
// subgraph.
struct ImageSegmenterOutputs {
  std::optional<std::vector<Source<Image>>> segmented_masks;
  std::optional<std::vector<Source<Image>>> confidence_masks;
  std::optional<Source<Image>> category_mask;
  // The same as the input image, mainly used for live stream mode.
  std::optional<Source<std::vector<float>>> quality_scores;
  Source<Image> image;
};

// Struct holding the image and input tensors after image preprocessing and
// transferred to the requested device.
struct ImageAndTensorsOnDevice {
  Source<Image> image;
  Source<std::vector<Tensor>> tensors;
};

}  // namespace

absl::Status SanityCheckOptions(const ImageSegmenterGraphOptions& options) {
  // TODO: remove deprecated output type support.
  if (options.segmenter_options().has_output_type() &&
      options.segmenter_options().output_type() ==
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
  MP_ASSIGN_OR_RETURN(absl::string_view labels_file,
                      metadata_extractor.GetAssociatedFile(labels_filename));
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_AXIS_LABELS,
          locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    MP_ASSIGN_OR_RETURN(
        display_names_file,
        metadata_extractor.GetAssociatedFile(display_names_filename));
  }
  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::Status ConfigureTensorsToSegmentationCalculator(
    const ImageSegmenterGraphOptions& segmenter_option,
    const core::ModelResources& model_resources,
    TensorsToSegmentationCalculatorOptions* options) {
  // Set default activation function NONE
  options->mutable_segmenter_options()->CopyFrom(
      segmenter_option.segmenter_options());
  // Find the custom metadata of ImageSegmenterOptions type in model metadata.
  const auto* metadata_extractor = model_resources.GetMetadataExtractor();
  bool found_activation_in_metadata = false;
  if (metadata_extractor->GetCustomMetadataList() != nullptr &&
      metadata_extractor->GetCustomMetadataList()->size() > 0) {
    for (const auto& custom_metadata :
         *metadata_extractor->GetCustomMetadataList()) {
      if (custom_metadata->name()->str() == kSegmentationMetadataName) {
        found_activation_in_metadata = true;
        auto activation_fb =
            GetImageSegmenterOptions(custom_metadata->data()->data())
                ->activation();
        switch (activation_fb) {
          case Activation_NONE:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::NONE);
            break;
          case Activation_SIGMOID:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::SIGMOID);
            break;
          case Activation_SOFTMAX:
            options->mutable_segmenter_options()->set_activation(
                SegmenterOptions::SOFTMAX);
            break;
          default:
            return CreateStatusWithPayload(
                absl::StatusCode::kInvalidArgument,
                "Invalid activation type found in CustomMetadata of "
                "ImageSegmenterOptions type.");
        }
      }
    }
  }
  if (!found_activation_in_metadata) {
    ABSL_LOG(WARNING)
        << "No activation type is found in model metadata. Use NONE for "
           "ImageSegmenterGraph.";
  }
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Segmentation tflite models are assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  MP_ASSIGN_OR_RETURN(
      *options->mutable_label_items(),
      GetLabelItemsIfAny(
          *metadata_extractor,
          **metadata_extractor->GetOutputTensorMetadata()->crbegin(),
          segmenter_option.display_names_locale()));
  return absl::OkStatus();
}

// Get the output tensor from the tflite model of given model resources.
absl::StatusOr<const tflite::Tensor*> GetOutputTensor(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  const auto* output_tensor =
      (*primary_subgraph->tensors())[*(*primary_subgraph->outputs()).rbegin()];
  return output_tensor;
}

uint32_t GetOutputTensorsSize(const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  return primary_subgraph->outputs()->size();
}

// Get the input tensor from the tflite model of given model resources.
absl::StatusOr<const tflite::Tensor*> GetInputTensor(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  const auto* primary_subgraph = (*model.subgraphs())[0];
  const auto* input_tensor =
      (*primary_subgraph->tensors())[(*primary_subgraph->inputs())[0]];
  return input_tensor;
}

// Configure the ImageTransformationCalculator according to the input tensor.
void ConfigureImageTransformationCalculator(
    const tflite::Tensor& tflite_input_tensor,
    mediapipe::ImageTransformationCalculatorOptions& options) {
  options.set_output_height(tflite_input_tensor.shape()->data()[1]);
  options.set_output_width(tflite_input_tensor.shape()->data()[2]);
}

// Configure the TensorConverterCalculator to convert the image to tensor.
void ConfigureTensorConverterCalculator(
    const ImageTensorSpecs& image_tensor_specs,
    mediapipe::TensorConverterCalculatorOptions& options) {
  float mean = image_tensor_specs.normalization_options->mean_values[0];
  float std = image_tensor_specs.normalization_options->std_values[0];
  options.set_max_num_channels(4);
  options.mutable_output_tensor_float_range()->set_min((0.0f - mean) / std);
  options.mutable_output_tensor_float_range()->set_max((255.0f - mean) / std);
}

// Image preprocessing step to convert the given image to the input tensors for
// the tflite model.
absl::StatusOr<ImageAndTensorsOnDevice> ConvertImageToTensors(
    Source<Image> image_in, Source<NormalizedRect> norm_rect_in, bool use_gpu,
    const core::proto::BaseOptions& base_options, bool is_hair_segmentation,
    const core::ModelResources& model_resources, Graph& graph) {
  MP_ASSIGN_OR_RETURN(const tflite::Tensor* tflite_input_tensor,
                      GetInputTensor(model_resources));
  if (tflite_input_tensor->shape()->size() != 4) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expect segmentation model has input image tensor to "
                        "be 4 dims. Got input tensor with "
                        "dims: %d",
                        tflite_input_tensor->shape()->size()));
  }
  const int input_tensor_channel = tflite_input_tensor->shape()->data()[3];
  if (input_tensor_channel != 3 && input_tensor_channel != 4) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expect segmentation model has input image tensor with channels = 3 or "
        "4. Get "
        "channel = %d",
        tflite_input_tensor->shape()->data()[3]));
  } else if (input_tensor_channel == 3) {
    // ImagePreprocessingGraph is backed by ImageToTensorCalculator which only
    // supports Tensor with channel = 3.
    auto& preprocessing = graph.AddNode(
        "mediapipe.tasks.components.processors.ImagePreprocessingGraph");
    MP_RETURN_IF_ERROR(components::processors::ConfigureImagePreprocessingGraph(
        model_resources, use_gpu, base_options.gpu_origin(),
        &preprocessing.GetOptions<tasks::components::processors::proto::
                                      ImagePreprocessingGraphOptions>()));
    image_in >> preprocessing.In(kImageTag);
    norm_rect_in >> preprocessing.In(kNormRectTag);
    return {{preprocessing.Out(kImageTag).Cast<Image>(),
             preprocessing.Out(kTensorsTag).Cast<std::vector<Tensor>>()}};
  } else {
    // TODO Remove legacy preprocessing calculators.
    // For segmentation model with input Tensor with channel = 4, use legacy
    // TfLite preprocessing calculators

    // Upload image to GPU if requested to use gpu.
    auto& image_clone = graph.AddNode("ImageCloneCalculator");
    image_clone.GetOptions<mediapipe::ImageCloneCalculatorOptions>()
        .set_output_on_gpu(use_gpu);
    image_in >> image_clone.In("");
    Source<Image> image_on_device = image_clone.Out("").Cast<Image>();

    // Convert from Image to legacy ImageFrame or GpuBuffer.
    auto& from_image = graph.AddNode("FromImageCalculator");
    image_on_device >> from_image.In(kImageTag);
    Source<api2::AnyType> image_cpu_or_gpu =
        from_image.Out(use_gpu ? kImageGpuTag : kImageCpuTag);

    if (is_hair_segmentation) {
      auto& set_alpha = graph.AddNode("SetAlphaCalculator");
      set_alpha.GetOptions<mediapipe::SetAlphaCalculatorOptions>()
          .set_alpha_value(0);
      image_cpu_or_gpu >> set_alpha.In(use_gpu ? kImageGpuTag : kImageTag);
      image_cpu_or_gpu = set_alpha.Out(use_gpu ? kImageGpuTag : kImageTag);
    }

    // Resize the input image to the model input size.
    auto& image_transformation = graph.AddNode("ImageTransformationCalculator");
    ConfigureImageTransformationCalculator(
        *tflite_input_tensor,
        image_transformation
            .GetOptions<mediapipe::ImageTransformationCalculatorOptions>());
    const absl::string_view image_or_image_gpu_tag =
        use_gpu ? kImageGpuTag : kImageTag;
    image_cpu_or_gpu >> image_transformation.In(image_or_image_gpu_tag);
    auto transformed_image = image_transformation.Out(image_or_image_gpu_tag);

    // Convert image to mediapipe tensor.
    auto& tensor_converter = graph.AddNode("TensorConverterCalculator");
    MP_ASSIGN_OR_RETURN(auto image_tensor_specs,
                        vision::BuildInputImageTensorSpecs(model_resources));
    ConfigureTensorConverterCalculator(
        image_tensor_specs,
        tensor_converter
            .GetOptions<mediapipe::TensorConverterCalculatorOptions>());

    transformed_image >> tensor_converter.In(image_or_image_gpu_tag);
    auto tensors =
        tensor_converter.Out(kTensorsTag).Cast<std::vector<Tensor>>();

    return {{image_on_device, tensors}};
  }
}

// An "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph" performs
// semantic segmentation. The graph can output optional confidence masks if
// CONFIDENCE_MASKS is connected, and an optional category mask if CATEGORY_MASK
// is connected. At least one of CONFIDENCE_MASK, CONFIDENCE_MASKS and
// CATEGORY_MASK must be connected.
//
//  Two kinds of outputs for confidence mask are provided: CONFIDENCE_MASK and
//  CONFIDENCE_MASKS. Users can retrieve segmented mask of only particular
//  category/channel from CONFIDENCE_MASK, and users can also get all segmented
//  confidence masks from CONFIDENCE_MASKS.
// - Accepts CPU input images and outputs segmented masks on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform segmentation on.
//   NORM_RECT - NormalizedRect @Optional
//     Describes image rotation and region of image to perform detection
//     on.
//     @Optional: rect covering the whole image is used if not specified.
//   OUTPUT_SIZE - std::pair<int, int> @Optional
//     The output size of the mask, in width and height. If not specified, the
//     output size of the input image is used.
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
//   calculator: "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph"
//   input_stream: "IMAGE:image"
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
//         activation: SOFTMAX
//       }
//     }
//   }
// }
class ImageSegmenterGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    MP_ASSIGN_OR_RETURN(const auto* model_resources,
                        CreateModelResources<ImageSegmenterGraphOptions>(sc));
    Graph graph;
    const auto& options = sc->Options<ImageSegmenterGraphOptions>();
    // TODO: remove deprecated output type support.
    if (!options.segmenter_options().has_output_type()) {
      MP_RETURN_IF_ERROR(SanityCheck(sc));
    }
    std::optional<Source<std::pair<int, int>>> output_size;
    if (HasInput(sc->OriginalNode(), kOutputSizeTag)) {
      output_size = graph.In(kOutputSizeTag).Cast<std::pair<int, int>>();
    }
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildSegmentationTask(
            options, *model_resources, graph[Input<Image>(kImageTag)],
            graph[Input<NormalizedRect>::Optional(kNormRectTag)], output_size,
            graph));

    // TODO: remove deprecated output type support.
    if (options.segmenter_options().has_output_type()) {
      auto& merge_images_to_vector =
          graph.AddNode("MergeImagesToVectorCalculator");
      for (int i = 0; i < output_streams.segmented_masks->size(); ++i) {
        output_streams.segmented_masks->at(i) >>
            merge_images_to_vector[Input<Image>::Multiple("")][i];
        output_streams.segmented_masks->at(i) >>
            graph[Output<Image>::Multiple(kSegmentationTag)][i];
      }
      merge_images_to_vector.Out("") >>
          graph[Output<std::vector<Image>>(kGroupedSegmentationTag)];
    } else {
      if (output_streams.confidence_masks) {
        auto& merge_images_to_vector =
            graph.AddNode("MergeImagesToVectorCalculator");
        for (int i = 0; i < output_streams.confidence_masks->size(); ++i) {
          output_streams.confidence_masks->at(i) >>
              merge_images_to_vector[Input<Image>::Multiple("")][i];
          output_streams.confidence_masks->at(i) >>
              graph[Output<Image>::Multiple(kConfidenceMaskTag)][i];
        }
        merge_images_to_vector.Out("") >>
            graph[Output<std::vector<Image>>::Optional(kConfidenceMasksTag)];
      }
      if (output_streams.category_mask) {
        *output_streams.category_mask >> graph[Output<Image>(kCategoryMaskTag)];
      }
    }
    if (output_streams.quality_scores) {
      *output_streams.quality_scores >>
          graph[Output<std::vector<float>>::Optional(kQualityScoresTag)];
    }
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  absl::Status SanityCheck(mediapipe::SubgraphContext* sc) {
    const auto& node = sc->OriginalNode();
    output_confidence_masks_ = HasOutput(node, kConfidenceMaskTag) ||
                               HasOutput(node, kConfidenceMasksTag);
    output_category_mask_ = HasOutput(node, kCategoryMaskTag);
    if (!output_confidence_masks_ && !output_category_mask_) {
      return absl::InvalidArgumentError(
          "At least one of CONFIDENCE_MASK, CONFIDENCE_MASKS and CATEGORY_MASK "
          "must be connected.");
    }
    return absl::OkStatus();
  }

  // Adds a mediapipe image segmentation task pipeline graph into the provided
  // builder::Graph instance. The segmentation pipeline takes images
  // (mediapipe::Image) as the input and returns segmented image mask as output.
  //
  // task_options: the mediapipe tasks ImageSegmenterGraphOptions proto.
  // model_resources: the ModelSources object initialized from a segmentation
  // model file with model metadata.
  // image_in: (mediapipe::Image) stream to run segmentation on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ImageSegmenterOutputs> BuildSegmentationTask(
      const ImageSegmenterGraphOptions& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Source<NormalizedRect> norm_rect_in,
      std::optional<Source<std::pair<int, int>>> output_size, Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));

    // Adds preprocessing calculators and connects them to the graph input image
    // stream.
    bool use_gpu =
        components::processors::DetermineImagePreprocessingGpuBackend(
            task_options.base_options().acceleration());

    // Adds segmentation calculators for output streams. Add this calculator
    // first to get the labels.
    auto& tensor_to_images =
        graph.AddNode("mediapipe.tasks.TensorsToSegmentationCalculator");
    RET_CHECK_OK(ConfigureTensorsToSegmentationCalculator(
        task_options, model_resources,
        &tensor_to_images
             .GetOptions<TensorsToSegmentationCalculatorOptions>()));
    const auto& tensor_to_images_options =
        tensor_to_images.GetOptions<TensorsToSegmentationCalculatorOptions>();

    // TODO: remove special logic for hair segmentation model.
    // The alpha channel of hair segmentation model indicates the interested
    // area. The model was designed for live stream mode, so that the mask of
    // previous frame is used as the indicator for the next frame. For the first
    // frame, it expects the alpha channel to be empty. To consolidate IMAGE,
    // VIDEO and LIVE_STREAM mode in mediapipe tasks, here we forcely set the
    // alpha channel to be empty if we find the model is the hair segmentation
    // model.
    bool is_hair_segmentation = false;
    if (tensor_to_images_options.label_items_size() == 2 &&
        tensor_to_images_options.label_items().at(1).name() == "hair") {
      is_hair_segmentation = true;
    }

    MP_ASSIGN_OR_RETURN(
        auto image_and_tensors,
        ConvertImageToTensors(image_in, norm_rect_in, use_gpu,
                              task_options.base_options(), is_hair_segmentation,
                              model_resources, graph));
    // Adds inference subgraph and connects its input stream to the output
    // tensors produced by the ImageToTensorCalculator.
    auto& inference = AddInference(
        model_resources, task_options.base_options().acceleration(), graph);
    image_and_tensors.tensors >> inference.In(kTensorsTag);
    inference.Out(kTensorsTag) >> tensor_to_images.In(kTensorsTag);

    if (output_size.has_value()) {
      *output_size >> tensor_to_images.In(kOutputSizeTag);
    } else {
      // Adds image property calculator for output size.
      auto& image_properties = graph.AddNode("ImagePropertiesCalculator");
      image_in >> image_properties.In(kImageTag);
      image_properties.Out(kSizeTag) >> tensor_to_images.In(kOutputSizeTag);
    }

    // Exports multiple segmented masks.
    // TODO: remove deprecated output type support.
    if (task_options.segmenter_options().has_output_type()) {
      std::vector<Source<Image>> segmented_masks;
      if (task_options.segmenter_options().output_type() ==
          SegmenterOptions::CATEGORY_MASK) {
        segmented_masks.push_back(
            Source<Image>(tensor_to_images[Output<Image>(kSegmentationTag)]));
      } else {
        MP_ASSIGN_OR_RETURN(const tflite::Tensor* output_tensor,
                            GetOutputTensor(model_resources));
        int segmentation_streams_num = *output_tensor->shape()->rbegin();
        for (int i = 0; i < segmentation_streams_num; ++i) {
          segmented_masks.push_back(Source<Image>(
              tensor_to_images[Output<Image>::Multiple(kSegmentationTag)][i]));
        }
      }
      auto quality_scores =
          tensor_to_images[Output<std::vector<float>>(kQualityScoresTag)];
      return ImageSegmenterOutputs{/*segmented_masks=*/segmented_masks,
                                   /*confidence_masks=*/std::nullopt,
                                   /*category_mask=*/std::nullopt,
                                   /*quality_scores=*/quality_scores,
                                   /*image=*/image_and_tensors.image};
    } else {
      std::optional<std::vector<Source<Image>>> confidence_masks;
      if (output_confidence_masks_) {
        MP_ASSIGN_OR_RETURN(const tflite::Tensor* output_tensor,
                            GetOutputTensor(model_resources));
        int segmentation_streams_num = *output_tensor->shape()->rbegin();
        confidence_masks = std::vector<Source<Image>>();
        confidence_masks->reserve(segmentation_streams_num);
        for (int i = 0; i < segmentation_streams_num; ++i) {
          confidence_masks->push_back(Source<Image>(
              tensor_to_images[Output<Image>::Multiple(kConfidenceMaskTag)]
                              [i]));
        }
      }
      std::optional<Source<Image>> category_mask;
      if (output_category_mask_) {
        category_mask = tensor_to_images[Output<Image>(kCategoryMaskTag)];
      }
      auto quality_scores =
          tensor_to_images[Output<std::vector<float>>(kQualityScoresTag)];
      return ImageSegmenterOutputs{/*segmented_masks=*/std::nullopt,
                                   /*confidence_masks=*/confidence_masks,
                                   /*category_mask=*/category_mask,
                                   /*quality_scores=*/quality_scores,
                                   /*image=*/image_and_tensors.image};
    }
  }

  bool output_confidence_masks_ = false;
  bool output_category_mask_ = false;
};

REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::vision::image_segmenter::ImageSegmenterGraph);

}  // namespace image_segmenter
}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
