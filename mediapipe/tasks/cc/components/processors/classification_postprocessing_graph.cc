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
#include "mediapipe/tasks/cc/components/processors/classification_postprocessing_graph.h"

#include <stdint.h>

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/core/split_vector_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensors_to_classification_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/calculators/classification_aggregation_calculator.pb.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_calculator.pb.h"
#include "mediapipe/tasks/cc/components/calculators/score_calibration_utils.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/classification_postprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/classifier_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace processors {

namespace {

using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::Timestamp;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::tflite::ProcessUnit;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64_t, ::mediapipe::LabelMapItem>;
using TensorsSource = mediapipe::api2::builder::Source<std::vector<Tensor>>;

constexpr float kDefaultScoreThreshold = std::numeric_limits<float>::lowest();

constexpr char kCalibratedScoresTag[] = "CALIBRATED_SCORES";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kScoresTag[] = "SCORES";
constexpr char kTensorsTag[] = "TENSORS";
constexpr char kTimestampsTag[] = "TIMESTAMPS";
constexpr char kTimestampedClassificationsTag[] = "TIMESTAMPED_CLASSIFICATIONS";

// Struct holding the different output streams produced by the graph.
struct ClassificationPostprocessingOutputStreams {
  Source<ClassificationResult> classifications;
  Source<std::vector<ClassificationResult>> timestamped_classifications;
};

// Performs sanity checks on provided ClassifierOptions.
absl::Status SanityCheckClassifierOptions(
    const proto::ClassifierOptions& options) {
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0.",
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

struct ClassificationHeadsProperties {
  int num_heads;
  bool quantized;
};

// Identifies the number of classification heads and whether they are quantized
// or not.
absl::StatusOr<ClassificationHeadsProperties> GetClassificationHeadsProperties(
    const ModelResources& model_resources) {
  const tflite::Model& model = *model_resources.GetTfLiteModel();
  if (model.subgraphs()->size() != 1) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Classification tflite models are "
                                   "assumed to have a single subgraph.",
                                   MediaPipeTasksStatus::kInvalidArgumentError);
  }
  const auto* primary_subgraph = (*model.subgraphs())[0];
  int num_output_tensors = primary_subgraph->outputs()->size();
  // Sanity check tensor types and check if model outputs are quantized or not.
  int num_quantized_tensors = 0;
  for (int i = 0; i < num_output_tensors; ++i) {
    const auto* tensor =
        primary_subgraph->tensors()->Get(primary_subgraph->outputs()->Get(i));
    if (tensor->type() != tflite::TensorType_FLOAT32 &&
        tensor->type() != tflite::TensorType_UINT8 &&
        tensor->type() != tflite::TensorType_BOOL) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Expected output tensor at index %d to have type "
                          "UINT8 or FLOAT32 or BOOL, found %s instead.",
                          i, tflite::EnumNameTensorType(tensor->type())),
          MediaPipeTasksStatus::kInvalidOutputTensorTypeError);
    }
    if (tensor->type() == tflite::TensorType_UINT8 ||
        tensor->type() == tflite::TensorType_BOOL) {
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
  return ClassificationHeadsProperties{
      /* num_heads= */ num_output_tensors,
      /* quantized= */ num_quantized_tensors > 0};
}

// Builds the label map from the tensor metadata, if available.
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

// Gets the score threshold from metadata, if any. Returns
// kDefaultScoreThreshold otherwise.
absl::StatusOr<float> GetScoreThreshold(
    const ModelMetadataExtractor& metadata_extractor,
    const TensorMetadata& tensor_metadata) {
  MP_ASSIGN_OR_RETURN(const ProcessUnit* score_thresholding_process_unit,
                      metadata_extractor.FindFirstProcessUnit(
                          tensor_metadata,
                          tflite::ProcessUnitOptions_ScoreThresholdingOptions));
  if (score_thresholding_process_unit == nullptr) {
    return kDefaultScoreThreshold;
  }
  return score_thresholding_process_unit->options_as_ScoreThresholdingOptions()
      ->global_score_threshold();
}

// Gets the category allowlist or denylist (if any) as a set of indices.
absl::StatusOr<absl::flat_hash_set<int>> GetAllowOrDenyCategoryIndicesIfAny(
    const proto::ClassifierOptions& options, const LabelItems& label_items) {
  absl::flat_hash_set<int> category_indices;
  // Exit early if no denylist/allowlist.
  if (options.category_denylist_size() == 0 &&
      options.category_allowlist_size() == 0) {
    return category_indices;
  }
  if (label_items.empty()) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Using `category_allowlist` or `category_denylist` requires labels to "
        "be present in the TFLite Model Metadata but none was found.",
        MediaPipeTasksStatus::kMetadataMissingLabelsError);
  }
  const auto& category_list = options.category_allowlist_size() > 0
                                  ? options.category_allowlist()
                                  : options.category_denylist();
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

absl::Status ConfigureScoreCalibrationIfAny(
    const ModelMetadataExtractor& metadata_extractor, int tensor_index,
    proto::ClassificationPostprocessingGraphOptions* options) {
  const auto* tensor_metadata =
      metadata_extractor.GetOutputTensorMetadata(tensor_index);
  if (tensor_metadata == nullptr) {
    return absl::OkStatus();
  }
  // Get ScoreCalibrationOptions, if any.
  MP_ASSIGN_OR_RETURN(const ProcessUnit* score_calibration_process_unit,
                      metadata_extractor.FindFirstProcessUnit(
                          *tensor_metadata,
                          tflite::ProcessUnitOptions_ScoreCalibrationOptions));
  if (score_calibration_process_unit == nullptr) {
    return absl::OkStatus();
  }
  auto* score_calibration_options =
      score_calibration_process_unit->options_as_ScoreCalibrationOptions();
  // Get corresponding AssociatedFile.
  auto score_calibration_filename =
      metadata_extractor.FindFirstAssociatedFileName(
          *tensor_metadata,
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
  ScoreCalibrationCalculatorOptions calculator_options;
  MP_RETURN_IF_ERROR(ConfigureScoreCalibration(
      score_calibration_options->score_transformation(),
      score_calibration_options->default_score(), score_calibration_file,
      &calculator_options));
  (*options->mutable_score_calibration_options())[tensor_index] =
      calculator_options;
  return absl::OkStatus();
}

void ConfigureClassificationAggregationCalculator(
    const ModelMetadataExtractor& metadata_extractor,
    mediapipe::ClassificationAggregationCalculatorOptions* options) {
  auto* output_tensors_metadata = metadata_extractor.GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr) {
    return;
  }
  for (const auto metadata : *output_tensors_metadata) {
    options->add_head_names(metadata->name()->str());
  }
}

}  // namespace

// Fills in the TensorsToClassificationCalculatorOptions based on the
// classifier options and the (optional) output tensor metadata.
absl::Status ConfigureTensorsToClassificationCalculator(
    const proto::ClassifierOptions& options,
    const ModelMetadataExtractor& metadata_extractor, int tensor_index,
    TensorsToClassificationCalculatorOptions* calculator_options) {
  const auto* tensor_metadata =
      metadata_extractor.GetOutputTensorMetadata(tensor_index);

  // Extract label map and score threshold from metadata, if available. Those
  // are optional for classification models.
  LabelItems label_items;
  float score_threshold = kDefaultScoreThreshold;
  if (tensor_metadata != nullptr) {
    MP_ASSIGN_OR_RETURN(label_items,
                        GetLabelItemsIfAny(metadata_extractor, *tensor_metadata,
                                           options.display_names_locale()));
    MP_ASSIGN_OR_RETURN(score_threshold, GetScoreThreshold(metadata_extractor,
                                                           *tensor_metadata));
  }
  // Allowlist / denylist.
  MP_ASSIGN_OR_RETURN(auto allow_or_deny_categories,
                      GetAllowOrDenyCategoryIndicesIfAny(options, label_items));
  if (!allow_or_deny_categories.empty()) {
    if (options.category_allowlist_size()) {
      calculator_options->mutable_allow_classes()->Assign(
          allow_or_deny_categories.begin(), allow_or_deny_categories.end());
    } else {
      calculator_options->mutable_ignore_classes()->Assign(
          allow_or_deny_categories.begin(), allow_or_deny_categories.end());
    }
  }
  // Score threshold.
  if (options.has_score_threshold()) {
    score_threshold = options.score_threshold();
  }
  calculator_options->set_min_score_threshold(score_threshold);
  // Number of results.
  if (options.max_results() > 0) {
    calculator_options->set_top_k(options.max_results());
  } else {
    // Setting to a negative value lets the calculator return all results.
    calculator_options->set_top_k(-1);
  }
  // Label map.
  *calculator_options->mutable_label_items() = std::move(label_items);
  // Always sort results.
  calculator_options->set_sort_by_descending_score(true);
  return absl::OkStatus();
}

absl::Status ConfigureClassificationPostprocessingGraph(
    const ModelResources& model_resources,
    const proto::ClassifierOptions& classifier_options,
    proto::ClassificationPostprocessingGraphOptions* options) {
  MP_RETURN_IF_ERROR(SanityCheckClassifierOptions(classifier_options));
  MP_ASSIGN_OR_RETURN(const auto heads_properties,
                      GetClassificationHeadsProperties(model_resources));
  for (int i = 0; i < heads_properties.num_heads; ++i) {
    MP_RETURN_IF_ERROR(ConfigureScoreCalibrationIfAny(
        *model_resources.GetMetadataExtractor(), i, options));
    MP_RETURN_IF_ERROR(ConfigureTensorsToClassificationCalculator(
        classifier_options, *model_resources.GetMetadataExtractor(), i,
        options->add_tensors_to_classifications_options()));
  }
  ConfigureClassificationAggregationCalculator(
      *model_resources.GetMetadataExtractor(),
      options->mutable_classification_aggregation_options());
  options->set_has_quantized_outputs(heads_properties.quantized);
  return absl::OkStatus();
}

// A "ClassificationPostprocessingGraph" converts raw tensors into
// ClassificationResult objects.
// - Accepts CPU input tensors.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of the timestamps that this calculator should aggregate.
//     This stream is optional: if provided then the TIMESTAMPED_CLASSIFICATIONS
//     output is used for results. Otherwise as no timestamp aggregation is
//     required the CLASSIFICATIONS output is used for results.
//
// Outputs:
//   CLASSIFICATIONS - ClassificationResult @Optional
//     The classification results aggregated by head. Must be connected if the
//     TIMESTAMPS input is not connected, as it signals that timestamp
//     aggregation is not required.
//   TIMESTAMPED_CLASSIFICATIONS - std::vector<ClassificationResult> @Optional
//     The classification result aggregated by timestamp, then by head. Must be
//     connected if the TIMESTAMPS input is connected, as it signals that
//     timestamp aggregation is required.
//
// The recommended way of using this graph is through the GraphBuilder API
// using the 'ConfigureClassificationPostprocessingGraph()' function. See header
// file for more details.
class ClassificationPostprocessingGraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    MP_ASSIGN_OR_RETURN(
        auto output_streams,
        BuildClassificationPostprocessing(
            sc->Options<proto::ClassificationPostprocessingGraphOptions>(),
            graph[Input<std::vector<Tensor>>(kTensorsTag)],
            graph[Input<std::vector<Timestamp>>(kTimestampsTag)], graph));
    output_streams.classifications >>
        graph[Output<ClassificationResult>(kClassificationsTag)];
    output_streams.timestamped_classifications >>
        graph[Output<std::vector<ClassificationResult>>(
            kTimestampedClassificationsTag)];
    return graph.GetConfig();
  }

 private:
  // Adds an on-device classification postprocessing graph into the provided
  // builder::Graph instance. The classification postprocessing graph takes
  // tensors (std::vector<mediapipe::Tensor>) and optional timestamps
  // (std::vector<Timestamp>) as input and returns two output streams:
  //  - classification results aggregated by classifier head as a
  //  ClassificationResult proto, used when no timestamps are passed in
  //    the graph,
  //  - classification results aggregated by timestamp then by classifier head
  //    as a std::vector<ClassificationResult>, used when timestamps are passed
  //    in the graph.
  //
  // options: the on-device ClassificationPostprocessingGraphOptions.
  // tensors_in: (std::vector<mediapipe::Tensor>>) tensors to postprocess.
  // timestamps_in: (std::vector<mediapipe::Timestamp>) optional collection of
  //   timestamps that should be used to aggregate classification results.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<ClassificationPostprocessingOutputStreams>
  BuildClassificationPostprocessing(
      const proto::ClassificationPostprocessingGraphOptions& options,
      Source<std::vector<Tensor>> tensors_in,
      Source<std::vector<Timestamp>> timestamps_in, Graph& graph) {
    const int num_heads = options.tensors_to_classifications_options_size();

    // Sanity check.
    if (num_heads == 0) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "ClassificationPostprocessingOptions must contain at least one "
          "TensorsToClassificationCalculatorOptions.",
          MediaPipeTasksStatus::kInvalidArgumentError);
    }

    // If output tensors are quantized, they must be dequantized first.
    TensorsSource dequantized_tensors = tensors_in;
    if (options.has_quantized_outputs()) {
      GenericNode* tensors_dequantization_node =
          &graph.AddNode("TensorsDequantizationCalculator");
      tensors_in >> tensors_dequantization_node->In(kTensorsTag);
      dequantized_tensors = tensors_dequantization_node->Out(kTensorsTag)
                                .Cast<std::vector<Tensor>>();
    }

    // If there are multiple classification heads, the output tensors need to be
    // split.
    std::vector<TensorsSource> split_tensors;
    split_tensors.reserve(num_heads);
    if (num_heads > 1) {
      GenericNode* split_tensor_vector_node =
          &graph.AddNode("SplitTensorVectorCalculator");
      auto& split_tensor_vector_options =
          split_tensor_vector_node
              ->GetOptions<mediapipe::SplitVectorCalculatorOptions>();
      for (int i = 0; i < num_heads; ++i) {
        auto* range = split_tensor_vector_options.add_ranges();
        range->set_begin(i);
        range->set_end(i + 1);
        split_tensors.push_back(
            split_tensor_vector_node->Out(i).Cast<std::vector<Tensor>>());
      }
      dequantized_tensors >> split_tensor_vector_node->In(0);
    } else {
      split_tensors.emplace_back(dequantized_tensors);
    }

    // Adds score calibration for heads that specify it, if any.
    std::vector<TensorsSource> calibrated_tensors;
    calibrated_tensors.reserve(num_heads);
    for (int i = 0; i < num_heads; ++i) {
      if (options.score_calibration_options().contains(i)) {
        GenericNode* score_calibration_node =
            &graph.AddNode("ScoreCalibrationCalculator");
        score_calibration_node->GetOptions<ScoreCalibrationCalculatorOptions>()
            .CopyFrom(options.score_calibration_options().at(i));
        split_tensors[i] >> score_calibration_node->In(kScoresTag);
        calibrated_tensors.push_back(
            score_calibration_node->Out(kCalibratedScoresTag)
                .Cast<std::vector<Tensor>>());
      } else {
        calibrated_tensors.emplace_back(split_tensors[i]);
      }
    }

    // Adds a TensorsToClassificationCalculator for each head.
    std::vector<GenericNode*> tensors_to_classification_nodes;
    tensors_to_classification_nodes.reserve(num_heads);
    for (int i = 0; i < num_heads; ++i) {
      tensors_to_classification_nodes.emplace_back(
          &graph.AddNode("TensorsToClassificationCalculator"));
      tensors_to_classification_nodes.back()
          ->GetOptions<TensorsToClassificationCalculatorOptions>()
          .CopyFrom(options.tensors_to_classifications_options(i));
      calibrated_tensors[i] >>
          tensors_to_classification_nodes.back()->In(kTensorsTag);
    }

    // Aggregates Classifications into a single ClassificationResult.
    auto& result_aggregation =
        graph.AddNode("ClassificationAggregationCalculator");
    result_aggregation
        .GetOptions<mediapipe::ClassificationAggregationCalculatorOptions>()
        .CopyFrom(options.classification_aggregation_options());
    for (int i = 0; i < num_heads; ++i) {
      tensors_to_classification_nodes[i]->Out(kClassificationsTag) >>
          result_aggregation.In(
              absl::StrFormat("%s:%d", kClassificationsTag, i));
    }
    timestamps_in >> result_aggregation.In(kTimestampsTag);

    // Connects output.
    ClassificationPostprocessingOutputStreams output_streams{
        /*classifications=*/
        result_aggregation[Output<ClassificationResult>(kClassificationsTag)],
        /*timestamped_classifications=*/
        result_aggregation[Output<std::vector<ClassificationResult>>(
            kTimestampedClassificationsTag)]};
    return output_streams;
  }
};

// REGISTER_MEDIAPIPE_GRAPH argument has to fit on one line to work properly.
// clang-format off
REGISTER_MEDIAPIPE_GRAPH(
  ::mediapipe::tasks::components::processors::ClassificationPostprocessingGraph); // NOLINT
// clang-format on

}  // namespace processors
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
