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

#include "mediapipe/tasks/cc/components/processors/detection_postprocessing_graph.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flexbuffers.h"
#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_detections_calculator.pb.h"
#include "mediapipe/calculators/tflite/ssd_anchors_calculator.pb.h"
#include "mediapipe/calculators/util/detection_label_id_to_text_calculator.pb.h"
#include "mediapipe/calculators/util/non_max_suppression_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/object_detection/anchor.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_utils.h"
#include "mediapipe/tasks/cc/components/processors/proto/detection_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/detector_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/tasks/metadata/object_detector_metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {

namespace {

using ::flatbuffers::Offset;
using ::flatbuffers::Vector;
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
using LabelItems = mediapipe::proto_ns::Map<int64_t, ::mediapipe::LabelMapItem>;
using TensorsSource =
    mediapipe::api2::builder::Source<std::vector<mediapipe::Tensor>>;

constexpr int kInModelNmsDefaultLocationsIndex = 0;
constexpr int kInModelNmsDefaultCategoriesIndex = 1;
constexpr int kInModelNmsDefaultScoresIndex = 2;
constexpr int kInModelNmsDefaultNumResultsIndex = 3;

constexpr int kOutModelNmsDefaultLocationsIndex = 0;
constexpr int kOutModelNmsDefaultScoresIndex = 1;

constexpr float kDefaultScoreThreshold = std::numeric_limits<float>::lowest();

constexpr absl::string_view kLocationTensorName = "location";
constexpr absl::string_view kCategoryTensorName = "category";
constexpr absl::string_view kScoreTensorName = "score";
constexpr absl::string_view kNumberOfDetectionsTensorName =
    "number of detections";
constexpr absl::string_view kDetectorMetadataName = "DETECTOR_METADATA";
constexpr absl::string_view kCalibratedScoresTag = "CALIBRATED_SCORES";
constexpr absl::string_view kDetectionsTag = "DETECTIONS";
constexpr absl::string_view kIndicesTag = "INDICES";
constexpr absl::string_view kScoresTag = "SCORES";
constexpr absl::string_view kTensorsTag = "TENSORS";
constexpr absl::string_view kAnchorsTag = "ANCHORS";
constexpr absl::string_view kDetectionPostProcessOpName =
    "TFLite_Detection_PostProcess";

// Struct holding the different output streams produced by the graph.
struct DetectionPostprocessingOutputStreams {
  Source<std::vector<Detection>> detections;
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
  // Score calibration options, if any.
  std::optional<ScoreCalibrationCalculatorOptions> score_calibration_options;
};

absl::Status SanityCheckOptions(const proto::DetectorOptions& options) {
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
    const TensorMetadata& tensor_metadata,
    tflite::AssociatedFileType associated_file_type, absl::string_view locale) {
  const std::string labels_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(tensor_metadata,
                                                          associated_file_type);
  if (labels_filename.empty()) {
    LabelItems empty_label_items;
    return empty_label_items;
  }
  MP_ASSIGN_OR_RETURN(absl::string_view labels_file,
                      metadata_extractor.GetAssociatedFile(labels_filename));
  const std::string display_names_filename =
      ModelMetadataExtractor::FindFirstAssociatedFileName(
          tensor_metadata, associated_file_type, locale);
  absl::string_view display_names_file;
  if (!display_names_filename.empty()) {
    MP_ASSIGN_OR_RETURN(
        display_names_file,
        metadata_extractor.GetAssociatedFile(display_names_filename));
  }
  return mediapipe::BuildLabelMapFromFiles(labels_file, display_names_file);
}

absl::StatusOr<float> GetScoreThreshold(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata) {
  MP_ASSIGN_OR_RETURN(
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
    const proto::DetectorOptions& config, const LabelItems& label_items) {
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

absl::StatusOr<std::optional<ScoreCalibrationCalculatorOptions>>
GetScoreCalibrationOptionsIfAny(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata) {
  // Get ScoreCalibrationOptions, if any.
  MP_ASSIGN_OR_RETURN(
      const ProcessUnit* score_calibration_process_unit,
      metadata_extractor.FindFirstProcessUnit(
          tensor_metadata, tflite::ProcessUnitOptions_ScoreCalibrationOptions));
  if (score_calibration_process_unit == nullptr) {
    return std::nullopt;
  }
  auto* score_calibration_options =
      score_calibration_process_unit->options_as_ScoreCalibrationOptions();
  // Get corresponding AssociatedFile.
  auto score_calibration_filename =
      metadata_extractor.FindFirstAssociatedFileName(
          tensor_metadata,
          tflite::AssociatedFileType_TENSOR_AXIS_SCORE_CALIBRATION);
  if (score_calibration_filename.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kNotFound,
        "Found ScoreCalibrationOptions but missing required associated "
        "parameters file with type TENSOR_AXIS_SCORE_CALIBRATION.",
        MediaPipeTasksStatus::kMetadataAssociatedFileNotFoundError);
  }
  MP_ASSIGN_OR_RETURN(
      absl::string_view score_calibration_file,
      metadata_extractor.GetAssociatedFile(score_calibration_filename));
  ScoreCalibrationCalculatorOptions score_calibration_calculator_options;
  MP_RETURN_IF_ERROR(ConfigureScoreCalibration(
      score_calibration_options->score_transformation(),
      score_calibration_options->default_score(), score_calibration_file,
      &score_calibration_calculator_options));
  return score_calibration_calculator_options;
}

absl::StatusOr<std::vector<int>> GetOutputTensorIndices(
    const Vector<Offset<TensorMetadata>>* tensor_metadatas) {
  std::vector<int> output_indices;
  if (tensor_metadatas->size() == 4) {
    output_indices = {
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
        ABSL_LOG(WARNING) << absl::StrFormat(
            "You don't seem to be matching tensor names in metadata list. The "
            "tensor name \"%s\" at index %d in the model metadata doesn't "
            "match "
            "the available output names: [\"%s\", \"%s\", \"%s\", \"%s\"].",
            tensor_metadatas->Get(i)->name()->c_str(), i, kLocationTensorName,
            kCategoryTensorName, kScoreTensorName,
            kNumberOfDetectionsTensorName);
        output_indices = {
            kInModelNmsDefaultLocationsIndex, kInModelNmsDefaultCategoriesIndex,
            kInModelNmsDefaultScoresIndex, kInModelNmsDefaultNumResultsIndex};
        return output_indices;
      }
    }
  } else if (tensor_metadatas->size() == 2) {
    output_indices = {core::FindTensorIndexByMetadataName(tensor_metadatas,
                                                          kLocationTensorName),
                      core::FindTensorIndexByMetadataName(tensor_metadatas,
                                                          kScoreTensorName)};
    // location, score
    for (int i = 0; i < 2; i++) {
      int output_index = output_indices[i];
      // If tensor name is not found, set the default output indices.
      if (output_index == -1) {
        ABSL_LOG(WARNING) << absl::StrFormat(
            "You don't seem to be matching tensor names in metadata list. The "
            "tensor name \"%s\" at index %d in the model metadata doesn't "
            "match "
            "the available output names: [\"%s\", \"%s\"].",
            tensor_metadatas->Get(i)->name()->c_str(), i, kLocationTensorName,
            kScoreTensorName);
        output_indices = {kOutModelNmsDefaultLocationsIndex,
                          kOutModelNmsDefaultScoresIndex};
        return output_indices;
      }
    }
  } else {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected a model with 2 or 4 output tensors metadata, found %d.",
            tensor_metadatas->size()),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  return output_indices;
}

// Get the MaxClassesPerDetection from TFLite_Detection_PostProcess op, if the
// op is found in the tflite model.
int GetMaxClassesPerDetection(const tflite::Model& model) {
  int max_classes_per_detection = 1;
  auto op_code_it = std::find_if(
      model.operator_codes()->begin(), model.operator_codes()->end(),
      [](const auto& op_code) {
        return op_code->builtin_code() == tflite::BuiltinOperator_CUSTOM &&
               op_code->custom_code()->str() == kDetectionPostProcessOpName;
      });
  if (op_code_it == model.operator_codes()->end()) {
    return max_classes_per_detection;
  }
  const int detection_opcode_index =
      op_code_it - model.operator_codes()->begin();
  const auto& operators = *model.subgraphs()->Get(0)->operators();
  auto detection_op_it =
      std::find_if(operators.begin(), operators.end(),
                   [detection_opcode_index](const auto& op) {
                     return op->opcode_index() == detection_opcode_index;
                   });
  if (detection_op_it != operators.end()) {
    auto op_config =
        flexbuffers::GetRoot(detection_op_it->custom_options()->Data(),
                             detection_op_it->custom_options()->size())
            .AsMap();
    return op_config["max_classes_per_detection"].AsInt32();
  }
  return max_classes_per_detection;
}

// Builds PostProcessingSpecs from DetectorOptions and model metadata for
// configuring the post-processing calculators.
absl::StatusOr<PostProcessingSpecs> BuildPostProcessingSpecs(
    const proto::DetectorOptions& options, bool in_model_nms,
    const ModelMetadataExtractor* metadata_extractor) {
  const auto* output_tensors_metadata =
      metadata_extractor->GetOutputTensorMetadata();
  PostProcessingSpecs specs;
  specs.max_results = options.max_results();
  MP_ASSIGN_OR_RETURN(specs.output_tensor_indices,
                      GetOutputTensorIndices(output_tensors_metadata));
  // Extracts mandatory BoundingBoxProperties and performs sanity checks on the
  // fly.
  MP_ASSIGN_OR_RETURN(const BoundingBoxProperties* bounding_box_properties,
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
  // For models with in-model-nms, the label map is stored in the Category
  // tensor which use TENSOR_VALUE_LABELS. For models with out-of-model-nms, the
  // label map is stored in the Score tensor which use TENSOR_AXIS_LABELS.
  MP_ASSIGN_OR_RETURN(
      specs.label_items,
      GetLabelItemsIfAny(
          *metadata_extractor,
          *output_tensors_metadata->Get(specs.output_tensor_indices[1]),
          in_model_nms ? tflite::AssociatedFileType_TENSOR_VALUE_LABELS
                       : tflite::AssociatedFileType_TENSOR_AXIS_LABELS,
          options.display_names_locale()));
  // Obtains allow/deny categories.
  specs.is_allowlist = !options.category_allowlist().empty();
  MP_ASSIGN_OR_RETURN(
      specs.allow_or_deny_categories,
      GetAllowOrDenyCategoryIndicesIfAny(options, specs.label_items));

  // Sets score threshold.
  if (options.has_score_threshold()) {
    specs.score_threshold = options.score_threshold();
  } else {
    MP_ASSIGN_OR_RETURN(
        specs.score_threshold,
        GetScoreThreshold(
            *metadata_extractor,
            *output_tensors_metadata->Get(
                specs.output_tensor_indices
                    [in_model_nms ? kInModelNmsDefaultScoresIndex
                                  : kOutModelNmsDefaultScoresIndex])));
  }
  if (in_model_nms) {
    // Builds score calibration options (if available) from metadata.
    MP_ASSIGN_OR_RETURN(
        specs.score_calibration_options,
        GetScoreCalibrationOptionsIfAny(
            *metadata_extractor,
            *output_tensors_metadata->Get(
                specs.output_tensor_indices[kInModelNmsDefaultScoresIndex])));
  }
  return specs;
}

// Builds PostProcessingSpecs from DetectorOptions and model metadata for
// configuring the post-processing calculators for models with
// non-maximum-suppression.
absl::StatusOr<PostProcessingSpecs> BuildInModelNmsPostProcessingSpecs(
    const proto::DetectorOptions& options,
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
  return BuildPostProcessingSpecs(options, /*in_model_nms=*/true,
                                  metadata_extractor);
}

// Fills in the TensorsToDetectionsCalculatorOptions based on
// PostProcessingSpecs.
void ConfigureInModelNmsTensorsToDetectionsCalculator(
    const PostProcessingSpecs& specs, const tflite::Model& model,
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

  options->set_max_classes_per_detection(GetMaxClassesPerDetection(model));
}

// Builds PostProcessingSpecs from DetectorOptions and model metadata for
// configuring the post-processing calculators for models without
// non-maximum-suppression.
absl::StatusOr<PostProcessingSpecs> BuildOutModelNmsPostProcessingSpecs(
    const proto::DetectorOptions& options,
    const ModelMetadataExtractor* metadata_extractor) {
  // Checks output tensor metadata is present and consistent with model.
  auto* output_tensors_metadata = metadata_extractor->GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr ||
      output_tensors_metadata->size() != 2) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Mismatch between number of output tensors (2) and "
                        "output tensors metadata (%d).",
                        output_tensors_metadata == nullptr
                            ? 0
                            : output_tensors_metadata->size()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }
  return BuildPostProcessingSpecs(options, /*in_model_nms=*/false,
                                  metadata_extractor);
}

// Configures the TensorsToDetectionCalculator for models without
// non-maximum-suppression in tflite model. The required config parameters are
// extracted from the ObjectDetectorMetadata
// (metadata/object_detector_metadata_schema.fbs).
absl::Status ConfigureOutModelNmsTensorsToDetectionsCalculator(
    const ModelMetadataExtractor* metadata_extractor,
    const PostProcessingSpecs& specs,
    mediapipe::TensorsToDetectionsCalculatorOptions* options) {
  bool found_detector_metadata = false;
  if (metadata_extractor->GetCustomMetadataList() != nullptr &&
      metadata_extractor->GetCustomMetadataList()->size() > 0) {
    for (const auto* custom_metadata :
         *metadata_extractor->GetCustomMetadataList()) {
      if (custom_metadata->name()->str() == kDetectorMetadataName) {
        found_detector_metadata = true;
        const auto* tensors_decoding_options =
            GetObjectDetectorOptions(custom_metadata->data()->data())
                ->tensors_decoding_options();
        // Here we don't set the max results for TensorsToDetectionsCalculator.
        // For models without nms, the results are filtered by max_results in
        // NonMaxSuppressionCalculator.
        options->set_num_classes(tensors_decoding_options->num_classes());
        options->set_num_boxes(tensors_decoding_options->num_boxes());
        options->set_num_coords(tensors_decoding_options->num_coords());
        options->set_keypoint_coord_offset(
            tensors_decoding_options->keypoint_coord_offset());
        options->set_num_keypoints(tensors_decoding_options->num_keypoints());
        options->set_num_values_per_keypoint(
            tensors_decoding_options->num_values_per_keypoint());
        options->set_x_scale(tensors_decoding_options->x_scale());
        options->set_y_scale(tensors_decoding_options->y_scale());
        options->set_w_scale(tensors_decoding_options->w_scale());
        options->set_h_scale(tensors_decoding_options->h_scale());
        options->set_apply_exponential_on_box_size(
            tensors_decoding_options->apply_exponential_on_box_size());
        options->set_sigmoid_score(tensors_decoding_options->sigmoid_score());
        break;
      }
    }
  }
  if (!found_detector_metadata) {
    return absl::InvalidArgumentError(
        "TensorsDecodingOptions is not found in the object detector "
        "metadata.");
  }
  // Options not configured through metadata.
  options->set_box_format(
      mediapipe::TensorsToDetectionsCalculatorOptions::YXHW);
  options->set_min_score_thresh(specs.score_threshold);
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
  tensor_mapping->set_scores_tensor_index(output_indices[1]);
  return absl::OkStatus();
}

// Configures the SsdAnchorsCalculator for models without
// non-maximum-suppression in tflite model. The required config parameters are
// extracted from the ObjectDetectorMetadata
// (metadata/object_detector_metadata_schema.fbs).
absl::Status ConfigureSsdAnchorsCalculator(
    const ModelMetadataExtractor* metadata_extractor,
    mediapipe::SsdAnchorsCalculatorOptions* options) {
  bool found_detector_metadata = false;
  if (metadata_extractor->GetCustomMetadataList() != nullptr &&
      metadata_extractor->GetCustomMetadataList()->size() > 0) {
    for (const auto* custom_metadata :
         *metadata_extractor->GetCustomMetadataList()) {
      if (custom_metadata->name()->str() == kDetectorMetadataName) {
        found_detector_metadata = true;
        const auto* ssd_anchors_options =
            GetObjectDetectorOptions(custom_metadata->data()->data())
                ->ssd_anchors_options();
        for (const auto* ssd_anchor :
             *ssd_anchors_options->fixed_anchors_schema()->anchors()) {
          auto* fixed_anchor = options->add_fixed_anchors();
          fixed_anchor->set_y_center(ssd_anchor->y_center());
          fixed_anchor->set_x_center(ssd_anchor->x_center());
          fixed_anchor->set_h(ssd_anchor->height());
          fixed_anchor->set_w(ssd_anchor->width());
        }
        break;
      }
    }
  }
  if (!found_detector_metadata) {
    return absl::InvalidArgumentError(
        "SsdAnchorsOptions is not found in the object detector "
        "metadata.");
  }
  return absl::OkStatus();
}

// Sets the default IoU-based non-maximum-suppression configs, and set the
// min_suppression_threshold and max_results for detection models without
// non-maximum-suppression.
void ConfigureNonMaxSuppressionCalculator(
    const proto::DetectorOptions& detector_options,
    mediapipe::NonMaxSuppressionCalculatorOptions* options) {
  options->set_min_suppression_threshold(
      detector_options.min_suppression_threshold());
  options->set_overlap_type(
      mediapipe::NonMaxSuppressionCalculatorOptions::INTERSECTION_OVER_UNION);
  options->set_algorithm(
      mediapipe::NonMaxSuppressionCalculatorOptions::DEFAULT);
  options->set_max_num_detections(detector_options.max_results());
  options->set_multiclass_nms(detector_options.multiclass_nms());
}

// Sets the labels from post PostProcessingSpecs.
void ConfigureDetectionLabelIdToTextCalculator(
    PostProcessingSpecs& specs,
    mediapipe::DetectionLabelIdToTextCalculatorOptions* options) {
  *options->mutable_label_items() = std::move(specs.label_items);
}

// Splits the vector of 4 output tensors from model inference and calibrate the
// score tensors according to the metadata, if any. Then concatenate the tensors
// back to a vector of 4 tensors.
absl::StatusOr<Source<std::vector<Tensor>>> CalibrateScores(
    Source<std::vector<Tensor>> model_output_tensors,
    const proto::DetectionPostprocessingGraphOptions& options, Graph& graph) {
  // Split tensors.
  auto* split_tensor_vector_node =
      &graph.AddNode("SplitTensorVectorCalculator");
  auto& split_tensor_vector_options =
      split_tensor_vector_node
          ->GetOptions<mediapipe::SplitVectorCalculatorOptions>();
  for (int i = 0; i < 4; ++i) {
    auto* range = split_tensor_vector_options.add_ranges();
    range->set_begin(i);
    range->set_end(i + 1);
  }
  model_output_tensors >> split_tensor_vector_node->In(0);

  // Add score calibration calculator.
  auto* score_calibration_node = &graph.AddNode("ScoreCalibrationCalculator");
  score_calibration_node->GetOptions<ScoreCalibrationCalculatorOptions>()
      .CopyFrom(options.score_calibration_options());
  const auto& tensor_mapping =
      options.tensors_to_detections_options().tensor_mapping();
  split_tensor_vector_node->Out(tensor_mapping.classes_tensor_index()) >>
      score_calibration_node->In(kIndicesTag);
  split_tensor_vector_node->Out(tensor_mapping.scores_tensor_index()) >>
      score_calibration_node->In(kScoresTag);

  // Re-concatenate tensors.
  auto* concatenate_tensor_vector_node =
      &graph.AddNode("ConcatenateTensorVectorCalculator");
  for (int i = 0; i < 4; ++i) {
    if (i == tensor_mapping.scores_tensor_index()) {
      score_calibration_node->Out(kCalibratedScoresTag) >>
          concatenate_tensor_vector_node->In(i);
    } else {
      split_tensor_vector_node->Out(i) >> concatenate_tensor_vector_node->In(i);
    }
  }
  model_output_tensors =
      concatenate_tensor_vector_node->Out(0).Cast<std::vector<Tensor>>();
  return model_output_tensors;
}

// Identifies whether or not the model has quantized outputs, and performs
// sanity checks.
absl::StatusOr<bool> HasQuantizedOutputs(
    const core::ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  // Model is checked to have single subgraph before.
  const auto* primary_subgraph = (*model.subgraphs())[0];
  int num_output_tensors = primary_subgraph->outputs()->size();
  // Sanity check tensor types and check if model outputs are quantized or not.
  int num_quantized_tensors = 0;
  for (int i = 0; i < num_output_tensors; ++i) {
    const auto* tensor =
        primary_subgraph->tensors()->Get(primary_subgraph->outputs()->Get(i));
    if (tensor->type() != tflite::TensorType_FLOAT32 &&
        tensor->type() != tflite::TensorType_UINT8) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Expected output tensor at index %d to have type "
                          "UINT8 or FLOAT32, found %s instead.",
                          i, tflite::EnumNameTensorType(tensor->type())),
          MediaPipeTasksStatus::kInvalidOutputTensorTypeError);
    }
    if (tensor->type() == tflite::TensorType_UINT8) {
      num_quantized_tensors++;
    }
  }
  if (num_quantized_tensors != num_output_tensors &&
      num_quantized_tensors != 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected either all or none of the output tensors to be "
            "quantized, but found %d quantized outputs for %d total outputs.",
            num_quantized_tensors, num_output_tensors),
        MediaPipeTasksStatus::kInvalidOutputTensorTypeError);
  }
  // Check if metadata is consistent with model topology.
  const auto* output_tensors_metadata =
      model_resources.GetMetadataExtractor()->GetOutputTensorMetadata();
  if (output_tensors_metadata != nullptr &&
      num_output_tensors != output_tensors_metadata->size()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Mismatch between number of output tensors (%d) and "
                        "output tensors metadata (%d).",
                        num_output_tensors, output_tensors_metadata->size()),
        MediaPipeTasksStatus::kMetadataInconsistencyError);
  }
  return num_quantized_tensors > 0;
}

}  // namespace

absl::Status ConfigureDetectionPostprocessingGraph(
    const tasks::core::ModelResources& model_resources,
    const proto::DetectorOptions& detector_options,
    proto::DetectionPostprocessingGraphOptions& options) {
  MP_RETURN_IF_ERROR(SanityCheckOptions(detector_options));
  const auto& model = *model_resources.GetTfLiteModel();
  bool in_model_nms = false;
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Expected a model with a single subgraph, found %d.",
                        model.subgraphs()->size()),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  if (model.subgraphs()->Get(0)->outputs()->size() == 2) {
    in_model_nms = false;
  } else if (model.subgraphs()->Get(0)->outputs()->size() == 4) {
    in_model_nms = true;
  } else {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Expected a model with 2 or 4 output tensors, found %d.",
            model.subgraphs()->Get(0)->outputs()->size()),
        MediaPipeTasksStatus::kInvalidArgumentError);
  }
  MP_ASSIGN_OR_RETURN(bool has_quantized_outputs,
                      HasQuantizedOutputs(model_resources));
  options.set_has_quantized_outputs(has_quantized_outputs);
  const ModelMetadataExtractor* metadata_extractor =
      model_resources.GetMetadataExtractor();
  if (in_model_nms) {
    MP_ASSIGN_OR_RETURN(auto post_processing_specs,
                        BuildInModelNmsPostProcessingSpecs(detector_options,
                                                           metadata_extractor));
    ConfigureInModelNmsTensorsToDetectionsCalculator(
        post_processing_specs, model,
        options.mutable_tensors_to_detections_options());
    ConfigureDetectionLabelIdToTextCalculator(
        post_processing_specs,
        options.mutable_detection_label_ids_to_text_options());
    if (post_processing_specs.score_calibration_options.has_value()) {
      *options.mutable_score_calibration_options() =
          std::move(*post_processing_specs.score_calibration_options);
    }
  } else {
    MP_ASSIGN_OR_RETURN(auto post_processing_specs,
                        BuildOutModelNmsPostProcessingSpecs(
                            detector_options, metadata_extractor));
    MP_RETURN_IF_ERROR(ConfigureOutModelNmsTensorsToDetectionsCalculator(
        metadata_extractor, post_processing_specs,
        options.mutable_tensors_to_detections_options()));
    MP_RETURN_IF_ERROR(ConfigureSsdAnchorsCalculator(
        metadata_extractor, options.mutable_ssd_anchors_options()));
    ConfigureNonMaxSuppressionCalculator(
        detector_options, options.mutable_non_max_suppression_options());
    ConfigureDetectionLabelIdToTextCalculator(
        post_processing_specs,
        options.mutable_detection_label_ids_to_text_options());
  }

  return absl::OkStatus();
}

// A DetectionPostprocessingGraph converts raw tensors into
// std::vector<Detection>.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator. The tensors vector could be
//     size 4 or size 2. Tensors vector of size 4 expects the tensors from the
//     models with DETECTION_POSTPROCESS ops in the tflite graph. Tensors vector
//     of size 2 expects the tensors from the models without the ops.
//   [1]:
//     https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/detection_postprocess.cc
// Outputs:
//   DETECTIONS - std::vector<Detection>
//     The postprocessed detection results.
//
// The recommended way of using this graph is through the GraphBuilder API
// using the 'ConfigureDetectionPostprocessingGraph()' function. See header
// file for more details.
class DetectionPostprocessingGraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildDetectionPostprocessing(
            *sc->MutableOptions<proto::DetectionPostprocessingGraphOptions>(),
            graph.In(kTensorsTag).Cast<std::vector<Tensor>>(), graph));
    output_streams.detections >>
        graph.Out(kDetectionsTag).Cast<std::vector<Detection>>();
    return graph.GetConfig();
  }

 private:
  // Adds an on-device detection postprocessing graph into the provided
  // builder::Graph instance. The detection postprocessing graph takes
  // tensors (std::vector<mediapipe::Tensor>) as input and returns one output
  // stream:
  //  - Detection results as a std::vector<Detection>.
  //
  // graph_options: the on-device DetectionPostprocessingGraphOptions.
  // tensors_in: (std::vector<mediapipe::Tensor>>) tensors to postprocess.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<DetectionPostprocessingOutputStreams>
  BuildDetectionPostprocessing(
      proto::DetectionPostprocessingGraphOptions& graph_options,
      Source<std::vector<Tensor>> tensors_in, Graph& graph) {
    Source<std::vector<Tensor>> tensors = tensors_in;
    if (graph_options.has_quantized_outputs()) {
      auto& tensors_dequantization_node =
          graph.AddNode("TensorsDequantizationCalculator");
      tensors_in >> tensors_dequantization_node.In(kTensorsTag);
      tensors = tensors_dequantization_node.Out(kTensorsTag)
                    .Cast<std::vector<Tensor>>();
    }
    std::optional<Source<std::vector<Detection>>> detections;
    if (!graph_options.has_non_max_suppression_options()) {
      // Calculators to perform score calibration, if specified in the options.
      if (graph_options.has_score_calibration_options()) {
        MP_ASSIGN_OR_RETURN(tensors,
                            CalibrateScores(tensors, graph_options, graph));
      }
      // Calculator to convert output tensors to a detection proto vector.
      auto& tensors_to_detections =
          graph.AddNode("TensorsToDetectionsCalculator");
      tensors_to_detections
          .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>()
          .Swap(graph_options.mutable_tensors_to_detections_options());
      tensors >> tensors_to_detections.In(kTensorsTag);
      detections = tensors_to_detections.Out(kDetectionsTag)
                       .Cast<std::vector<Detection>>();
    } else {
      // Generates a single side packet containing a vector of SSD anchors.
      auto& ssd_anchor = graph.AddNode("SsdAnchorsCalculator");
      ssd_anchor.GetOptions<mediapipe::SsdAnchorsCalculatorOptions>().Swap(
          graph_options.mutable_ssd_anchors_options());
      auto anchors =
          ssd_anchor.SideOut("").Cast<std::vector<mediapipe::Anchor>>();
      // Convert raw output tensors to detections.
      auto& tensors_to_detections =
          graph.AddNode("TensorsToDetectionsCalculator");
      tensors_to_detections
          .GetOptions<mediapipe::TensorsToDetectionsCalculatorOptions>()
          .Swap(graph_options.mutable_tensors_to_detections_options());
      anchors >> tensors_to_detections.SideIn(kAnchorsTag);
      tensors >> tensors_to_detections.In(kTensorsTag);
      detections = tensors_to_detections.Out(kDetectionsTag)
                       .Cast<std::vector<mediapipe::Detection>>();
      // Non maximum suppression removes redundant object detections.
      auto& non_maximum_suppression =
          graph.AddNode("NonMaxSuppressionCalculator");
      non_maximum_suppression
          .GetOptions<mediapipe::NonMaxSuppressionCalculatorOptions>()
          .Swap(graph_options.mutable_non_max_suppression_options());
      *detections >> non_maximum_suppression.In("");
      detections =
          non_maximum_suppression.Out("").Cast<std::vector<Detection>>();
    }

    // Calculator to assign detection labels.
    auto& detection_label_id_to_text =
        graph.AddNode("DetectionLabelIdToTextCalculator");
    detection_label_id_to_text
        .GetOptions<mediapipe::DetectionLabelIdToTextCalculatorOptions>()
        .Swap(graph_options.mutable_detection_label_ids_to_text_options());
    *detections >> detection_label_id_to_text.In("");
    return {
        {detection_label_id_to_text.Out("").Cast<std::vector<Detection>>()}};
  }
};

// REGISTER_MEDIAPIPE_GRAPH argument has to fit on one line to work properly.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::components::processors::DetectionPostprocessingGraph); // NOLINT
// clang-format on

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
