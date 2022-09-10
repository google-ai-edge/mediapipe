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
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/image_preprocessing.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/model_task_graph.h"
#include "mediapipe/tasks/cc/core/proto/inference_subgraph.pb.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/vision/object_detector/proto/object_detector_options.pb.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"

namespace mediapipe {
namespace tasks {
namespace vision {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::tflite::BoundingBoxProperties;
using ::tflite::ContentProperties;
using ::tflite::ContentProperties_BoundingBoxProperties;
using ::tflite::EnumNameContentProperties;
using ::tflite::ProcessUnit;
using ::tflite::ProcessUnitOptions_ScoreThresholdingOptions;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64, ::mediapipe::LabelMapItem>;
using ObjectDetectorOptionsProto =
    object_detector::proto::ObjectDetectorOptions;

constexpr int kDefaultLocationsIndex = 0;
constexpr int kDefaultCategoriesIndex = 1;
constexpr int kDefaultScoresIndex = 2;
constexpr int kDefaultNumResultsIndex = 3;

constexpr float kDefaultScoreThreshold = std::numeric_limits<float>::lowest();

constexpr char kLocationTensorName[] = "location";
constexpr char kCategoryTensorName[] = "category";
constexpr char kScoreTensorName[] = "score";
constexpr char kNumberOfDetectionsTensorName[] = "number of detections";

constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kMatrixTag[] = "MATRIX";
constexpr char kPixelDetectionsTag[] = "PIXEL_DETECTIONS";
constexpr char kProjectionMatrixTag[] = "PROJECTION_MATRIX";
constexpr char kTensorTag[] = "TENSORS";

// Struct holding the different output streams produced by the object detection
// subgraph.
struct ObjectDetectionOutputStreams {
  Source<std::vector<Detection>> detections;
  Source<Image> image;
};

// Parameters used for configuring the post-processing calculators.
struct PostProcessingSpecs {
  // The maximum number of detection results to return.
  int max_results;
  // Indices of the output tensors to match the output tensors to the correct
  // index order of the output tensors: [location, categories, scores,
  // num_detections].
  std::vector<int> output_tensor_indices;
  // For each pack of 4 coordinates returned by the model, this denotes the
  // order in which to get the left, top, right and bottom coordinates.
  std::vector<unsigned int> bounding_box_corners_order;
  // This is populated by reading the label files from the TFLite Model
  // Metadata: if no such files are available, this is left empty and the
  // ObjectDetector will only be able to populate the `index` field of the
  // detection results.
  LabelItems label_items;
  // Score threshold. Detections with a confidence below this value are
  // discarded. If none is provided via metadata or options, -FLT_MAX is set as
  // default value.
  float score_threshold;
  // Set of category indices to be allowed/denied.
  absl::flat_hash_set<int> allow_or_deny_categories;
  // Indicates `allow_or_deny_categories` is an allowlist or a denylist.
  bool is_allowlist;
  // TODO: Adds score calibration.
};

absl::Status SanityCheckOptions(const ObjectDetectorOptionsProto& options) {
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (options.category_allowlist_size() > 0 &&
      options.category_denylist_size() > 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "`category_allowlist` and `category_denylist` are mutually "
        "exclusive options.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::StatusOr<const BoundingBoxProperties*> GetBoundingBoxProperties(
    const TensorMetadata& tensor_metadata) {
  if (tensor_metadata.content() == nullptr ||
      tensor_metadata.content()->content_properties() == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected BoundingBoxProperties for tensor %s, found none.",
            tensor_metadata.name() ? tensor_metadata.name()->str() : "#0"),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  ContentProperties type = tensor_metadata.content()->content_properties_type();
  if (type != ContentProperties_BoundingBoxProperties) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected BoundingBoxProperties for tensor %s, found %s.",
            tensor_metadata.name() ? tensor_metadata.name()->str() : "#0",
            EnumNameContentProperties(type)),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  const BoundingBoxProperties* properties =
      tensor_metadata.content()->content_properties_as_BoundingBoxProperties();

  // Mobile SSD only supports "BOUNDARIES" bounding box type.
  if (properties->type() != tflite::BoundingBoxType_BOUNDARIES) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Mobile SSD only supports BoundingBoxType BOUNDARIES, found %s",
            tflite::EnumNameBoundingBoxType(properties->type())),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  // Mobile SSD only supports "RATIO" coordinates type.
  if (properties->coordinate_type() != tflite::CoordinateType_RATIO) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Mobile SSD only supports CoordinateType RATIO, found %s",
            tflite::EnumNameCoordinateType(properties->coordinate_type())),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  // Index is optional, but must contain 4 values if present.
  if (properties->index() != nullptr && properties->index()->size() != 4) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected BoundingBoxProperties index to contain 4 values, found "
            "%d",
            properties->index()->size()),
        MediaPipeTasksStatus::kMetadataInvalidContentPropertiesError);
  }

  return properties;
}

absl::StatusOr<LabelItems> GetLabelItemsIfAny(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata, absl::string_view locale) {
  const std::string labels_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_VALUE_LABELS);
  if (labels_filename.empty()) {
    LabelItems empty_label_items;
    return empty_label_items;
  }
  ASSIGN_OR_RETURN(absl::string_view labels_file,
                   metadata_extractor.GetAssociatedFile(labels_filename));
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, tflite::AssociatedFileType_TENSOR_VALUE_LABELS,
          locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    ASSIGN_OR_RETURN(display_names_file, metadata_extractor.GetAssociatedFile(
                                             display_names_filename));
  }
  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::StatusOr<float> GetScoreThreshold(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata) {
  ASSIGN_OR_RETURN(
      const ProcessUnit* score_thresholding_process_unit,
      metadata_extractor.FindFirstProcessUnit(
          tensor_metadata, ProcessUnitOptions_ScoreThresholdingOptions));
  if (score_thresholding_process_unit == nullptr) {
    return kDefaultScoreThreshold;
  }
  return score_thresholding_process_unit->options_as_ScoreThresholdingOptions()
      ->global_score_threshold();
}

absl::StatusOr<absl::flat_hash_set<int>> GetAllowOrDenyCategoryIndicesIfAny(
    const ObjectDetectorOptionsProto& config, const LabelItems& label_items) {
  absl::flat_hash_set<int> category_indices;
  // Exit early if no denylist/allowlist.
  if (config.category_denylist_size() == 0 &&
      config.category_allowlist_size() == 0) {
    return category_indices;
  }
  if (label_items.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Using `category_allowlist` or `category_denylist` requires "
        "labels to be present in the TFLite Model Metadata but none was found.",
        MediaPipeTasksStatus::kMetadataMissingLabelsError);
  }
  const auto& category_list = config.category_allowlist_size() > 0
                                  ? config.category_allowlist()
                                  : config.category_denylist();
  for (const auto& category_name : category_list) {
    int index = -1;
    for (int i = 0; i < label_items.size(); ++i) {
      if (label_items.at(i).name() == category_name) {
        index = i;
        break;
      }
    }
    // Ignores duplicate or unknown categories.
    if (index < 0) {
      continue;
    }
    category_indices.insert(index);
  }
  return category_indices;
}

std::vector<int> GetOutputTensorIndices(
    const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>>*
        tensor_metadatas) {
  std::vector<int> output_indices = {
      core::FindTensorIndexByMetadataName(tensor_metadatas,
                                          kLocationTensorName),
      core::FindTensorIndexByMetadataName(tensor_metadatas,
                                          kCategoryTensorName),
      core::FindTensorIndexByMetadataName(tensor_metadatas, kScoreTensorName),
      core::FindTensorIndexByMetadataName(tensor_metadatas,
                                          kNumberOfDetectionsTensorName)};
  // locations, categories, scores, and number of detections
  for (int i = 0; i < 4; i++) {
    int output_index = output_indices[i];
    // If tensor name is not found, set the default output indices.
    if (output_index == -1) {
      LOG(WARNING) << absl::StrFormat(
          "You don't seem to be matching tensor names in metadata list. The "
          "tensor name \"%s\" at index %d in the model metadata doesn't "
          "match "
          "the available output names: [\"%s\", \"%s\", \"%s\", \"%s\"].",
          tensor_metadatas->Get(i)->name()->c_str(), i, kLocationTensorName,
          kCategoryTensorName, kScoreTensorName, kNumberOfDetectionsTensorName);
      output_indices = {kDefaultLocationsIndex, kDefaultCategoriesIndex,
                        kDefaultScoresIndex, kDefaultNumResultsIndex};
      return output_indices;
    }
  }
  return output_indices;
}

// Builds PostProcessingSpecs from ObjectDetectorOptionsProto and model metadata
// for configuring the post-processing calculators.
absl::StatusOr<PostProcessingSpecs> BuildPostProcessingSpecs(
    const ObjectDetectorOptionsProto& options,
    const ModelMetadataExtractor* metadata_extractor) {
  // Checks output tensor metadata is present and consistent with model.
  auto* output_tensors_metadata = metadata_extractor->GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr ||
      output_tensors_metadata->size() != 4) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Mismatch between number of output tensors (4) and "
                        "output tensors metadata (%d).",
                        output_tensors_metadata == nullptr
                            ? 0
                            : output_tensors_metadata->size()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }
  PostProcessingSpecs specs;
  specs.max_results = options.max_results();
  specs.output_tensor_indices = GetOutputTensorIndices(output_tensors_metadata);
  // Extracts mandatory BoundingBoxProperties and performs sanity checks on the
  // fly.
  ASSIGN_OR_RETURN(const BoundingBoxProperties* bounding_box_properties,
                   GetBoundingBoxProperties(*output_tensors_metadata->Get(
                       specs.output_tensor_indices[0])));
  if (bounding_box_properties->index() == nullptr) {
    specs.bounding_box_corners_order = {0, 1, 2, 3};
  } else {
    auto bounding_box_index = bounding_box_properties->index();
    specs.bounding_box_corners_order = {
        bounding_box_index->Get(0),
        bounding_box_index->Get(1),
        bounding_box_index->Get(2),
        bounding_box_index->Get(3),
    };
  }
  // Builds label map (if available) from metadata.
  ASSIGN_OR_RETURN(specs.label_items,
                   GetLabelItemsIfAny(*metadata_extractor,
                                      *output_tensors_metadata->Get(
                                          specs.output_tensor_indices[1]),
                                      options.display_names_locale()));
  // Obtains allow/deny categories.
  specs.is_allowlist = !options.category_allowlist().empty();
  ASSIGN_OR_RETURN(
      specs.allow_or_deny_categories,
      GetAllowOrDenyCategoryIndicesIfAny(options, specs.label_items));
  // Sets score threshold.
  if (options.has_score_threshold()) {
    specs.score_threshold = options.score_threshold();
  } else {
    ASSIGN_OR_RETURN(specs.score_threshold,
                     GetScoreThreshold(*metadata_extractor,
                                       *output_tensors_metadata->Get(
                                           specs.output_tensor_indices[2])));
  }
  return specs;
}

// Fills in the TensorsToDetectionsCalculatorOptions based on
// PostProcessingSpecs.
void ConfigureTensorsToDetectionsCalculator(
    const PostProcessingSpecs& specs,
    mediapipe::TensorsToDetectionsCalculatorOptions* options) {
  options->set_num_classes(specs.label_items.size());
  options->set_num_coords(4);
  options->set_min_score_thresh(specs.score_threshold);
  if (specs.max_results != -1) {
    options->set_max_results(specs.max_results);
  }
  if (specs.is_allowlist) {
    options->mutable_allow_classes()->Assign(
        specs.allow_or_deny_categories.begin(),
        specs.allow_or_deny_categories.end());
  } else {
    options->mutable_ignore_classes()->Assign(
        specs.allow_or_deny_categories.begin(),
        specs.allow_or_deny_categories.end());
  }

  const auto& output_indices = specs.output_tensor_indices;
  // Assigns indices to each the model output tensor.
  auto* tensor_mapping = options->mutable_tensor_mapping();
  tensor_mapping->set_detections_tensor_index(output_indices[0]);
  tensor_mapping->set_classes_tensor_index(output_indices[1]);
  tensor_mapping->set_scores_tensor_index(output_indices[2]);
  tensor_mapping->set_num_detections_tensor_index(output_indices[3]);

  // Assigns the bounding box corner order.
  auto box_boundaries_indices = options->mutable_box_boundaries_indices();
  box_boundaries_indices->set_xmin(specs.bounding_box_corners_order[0]);
  box_boundaries_indices->set_ymin(specs.bounding_box_corners_order[1]);
  box_boundaries_indices->set_xmax(specs.bounding_box_corners_order[2]);
  box_boundaries_indices->set_ymax(specs.bounding_box_corners_order[3]);
}

}  // namespace

// A "mediapipe.tasks.vision.ObjectDetectorGraph" performs object detection.
// - Accepts CPU input images and outputs detections on CPU.
//
// Inputs:
//   IMAGE - Image
//     Image to perform detection on.
//
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     Detected objects with bounding box in pixel units.
//   IMAGE - mediapipe::Image
//     The image that object detection runs on.
//
// Example:
// node {
//   calculator: "mediapipe.tasks.vision.ObjectDetectorGraph"
//   input_stream: "IMAGE:image_in"
//   output_stream: "DETECTIONS:detections_out"
//   output_stream: "IMAGE:image_out"
//   options {
//     [mediapipe.tasks.vision.object_detector.proto.ObjectDetectorOptions.ext]
//     {
//       max_results: 4
//       score_threshold: 0.5
//       category_allowlist: "foo"
//       category_allowlist: "bar"
//     }
//   }
// }
class ObjectDetectorGraph : public core::ModelTaskGraph {
 public:
  absl::StatusOr<CalculatorGraphConfig> GetConfig(
      SubgraphContext* sc) override {
    ASSIGN_OR_RETURN(const auto* model_resources,
                     CreateModelResources<ObjectDetectorOptionsProto>(sc));
    Graph graph;
    ASSIGN_OR_RETURN(
        auto output_streams,
        BuildObjectDetectionTask(sc->Options<ObjectDetectorOptionsProto>(),
                                 *model_resources,
                                 graph[Input<Image>(kImageTag)], graph));
    output_streams.detections >>
        graph[Output<std::vector<Detection>>(kDetectionsTag)];
    output_streams.image >> graph[Output<Image>(kImageTag)];
    return graph.GetConfig();
  }

 private:
  // Adds a mediapipe object detection task graph into the provided
  // builder::Graph instance. The object detection task takes images
  // (mediapipe::Image) as the input and returns two output streams:
  //   - the detection results (std::vector<Detection>),
  //   - the processed image that has pixel data stored on the target storage
  //     (mediapipe::Image).
  //
  // task_options: the mediapipe tasks ObjectDetectorOptions proto.
  // model_resources: the ModelSources object initialized from an object
  // detection model file with model metadata.
  // image_in: (mediapipe::Image) stream to run object detection on.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ObjectDetectionOutputStreams> BuildObjectDetectionTask(
      const ObjectDetectorOptionsProto& task_options,
      const core::ModelResources& model_resources, Source<Image> image_in,
      Graph& graph) {
    MP_RETURN_IF_ERROR(SanityCheckOptions(task_options));
    auto metadata_extractor = model_resources.GetMetadataExtractor();
    // Checks that metadata is available.
    if (metadata_extractor->GetModelMetadata() == nullptr ||
        metadata_extractor->GetModelMetadata()->subgraph_metadata() ==
            nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Object detection models require TFLite Model Metadata but none was "
          "found",
          MediaPipeTasksStatus::kMetadataNotFoundError);
    }

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
    preprocessing.Out(kTensorTag) >> inference.In(kTensorTag);

    // Adds post processing calculators.
    ASSIGN_OR_RETURN(
        auto post_processing_specs,
        BuildPostProcessingSpecs(task_options, metadata_extractor));
    // Calculator to convert output tensors to a detection proto vector.
    // Connects TensorsToDetectionsCalculator's input stream to the output
    // tensors produced by the inference subgraph.
    auto& tensors_to_detections =
        graph.AddNode("TensorsToDetectionsCalculator");
    ConfigureTensorsToDetectionsCalculator(
        post_processing_specs,
        &tensors_to_detections
             .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>());
    inference.Out(kTensorTag) >> tensors_to_detections.In(kTensorTag);

    // Calculator to projects detections back to the original coordinate system.
    auto& detection_projection = graph.AddNode("DetectionProjectionCalculator");
    tensors_to_detections.Out(kDetectionsTag) >>
        detection_projection.In(kDetectionsTag);
    preprocessing.Out(kMatrixTag) >>
        detection_projection.In(kProjectionMatrixTag);

    // Calculator to convert relative detection bounding boxes to pixel
    // detection bounding boxes.
    auto& detection_transformation =
        graph.AddNode("DetectionTransformationCalculator");
    detection_projection.Out(kDetectionsTag) >>
        detection_transformation.In(kDetectionsTag);
    preprocessing.Out(kImageSizeTag) >>
        detection_transformation.In(kImageSizeTag);

    // Calculator to assign detection labels.
    auto& detection_label_id_to_text =
        graph.AddNode("DetectionLabelIdToTextCalculator");
    auto& detection_label_id_to_text_opts =
        detection_label_id_to_text
            .GetOptions<mediapipe::DetectionLabelIdToTextCalculatorOptions>();
    *detection_label_id_to_text_opts.mutable_label_items() =
        std::move(post_processing_specs.label_items);
    detection_transformation.Out(kPixelDetectionsTag) >>
        detection_label_id_to_text.In("");

    // Outputs the labeled detections and the processed image as the subgraph
    // output streams.
    return {{
        /* detections= */
        detection_label_id_to_text[Output<std::vector<Detection>>("")],
        /* image= */ preprocessing[Output<Image>(kImageTag)],
    }};
  }
};

REGISTER_MEDIAPIPE_GRAPH(::mediapipe::tasks::vision::ObjectDetectorGraph);

}  // namespace vision
}  // namespace tasks
}  // namespace mediapipe
