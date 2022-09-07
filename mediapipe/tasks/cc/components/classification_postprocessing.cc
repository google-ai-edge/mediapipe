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
#include "mediapipe/tasks/cc/components/classification_postprocessing.h"

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
#include "mediapipe/tasks/cc/components/classification_postprocessing_options.pb.h"
#include "mediapipe/tasks/cc/components/classifier_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"
#include "mediapipe/util/label_map.pb.h"
#include "mediapipe/util/label_map_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe {
namespace tasks {

namespace {

using ::mediapipe::Tensor;
using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::Timestamp;
using ::mediapipe::api2::builder::GenericNode;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::tflite::ProcessUnit;
using ::tflite::ProcessUnitOptions_ScoreThresholdingOptions;
using ::tflite::TensorMetadata;
using LabelItems = mediapipe::proto_ns::Map<int64, ::mediapipe::LabelMapItem>;

constexpr float kDefaultScoreThreshold = std::numeric_limits<float>::lowest();

constexpr char kTensorsTag[] = "TENSORS";
constexpr char kClassificationResultTag[] = "CLASSIFICATION_RESULT";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kTimestampsTag[] = "TIMESTAMPS";

// Performs sanity checks on provided ClassifierOptions.
absl::Status SanityCheckClassifierOptions(const ClassifierOptions& options) {
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

// Gets the score threshold from metadata, if any. Returns
// kDefaultScoreThreshold otherwise.
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

// Gets the category allowlist or denylist (if any) as a set of indices.
absl::StatusOr<absl::flat_hash_set<int>> GetAllowOrDenyCategoryIndicesIfAny(
    const ClassifierOptions& options, const LabelItems& label_items) {
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

// Fills in the TensorsToClassificationCalculatorOptions based on the classifier
// options and the (optional) output tensor metadata.
absl::Status ConfigureTensorsToClassificationCalculator(
    const ClassifierOptions& options,
    const ModelMetadataExtractor& metadata_extractor, int tensor_index,
    TensorsToClassificationCalculatorOptions* calculator_options) {
  const auto* tensor_metadata =
      metadata_extractor.GetOutputTensorMetadata(tensor_index);

  // Extract label map and score threshold from metadata, if available. Those
  // are optional for classification models.
  LabelItems label_items;
  float score_threshold = kDefaultScoreThreshold;
  if (tensor_metadata != nullptr) {
    ASSIGN_OR_RETURN(label_items,
                     GetLabelItemsIfAny(metadata_extractor, *tensor_metadata,
                                        options.display_names_locale()));
    ASSIGN_OR_RETURN(score_threshold,
                     GetScoreThreshold(metadata_extractor, *tensor_metadata));
  }
  // Allowlist / denylist.
  ASSIGN_OR_RETURN(auto allow_or_deny_categories,
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

void ConfigureClassificationAggregationCalculator(
    const ModelMetadataExtractor& metadata_extractor,
    ClassificationAggregationCalculatorOptions* options) {
  auto* output_tensors_metadata = metadata_extractor.GetOutputTensorMetadata();
  if (output_tensors_metadata == nullptr) {
    return;
  }
  for (const auto& metadata : *output_tensors_metadata) {
    options->add_head_names(metadata->name()->str());
  }
}

}  // namespace

absl::Status ConfigureClassificationPostprocessing(
    const ModelResources& model_resources,
    const ClassifierOptions& classifier_options,
    ClassificationPostprocessingOptions* options) {
  MP_RETURN_IF_ERROR(SanityCheckClassifierOptions(classifier_options));
  ASSIGN_OR_RETURN(const auto heads_properties,
                   GetClassificationHeadsProperties(model_resources));
  for (int i = 0; i < heads_properties.num_heads; ++i) {
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

// A "mediapipe.tasks.ClassificationPostprocessingSubgraph" converts raw
// tensors into ClassificationResult objects.
// - Accepts CPU input tensors.
//
// Inputs:
//   TENSORS - std::vector<Tensor>
//     The output tensors of an InferenceCalculator.
//   TIMESTAMPS - std::vector<Timestamp> @Optional
//     The collection of timestamps that a single ClassificationResult should
//     aggregate. This is mostly useful for classifiers working on time series,
//     e.g. audio or video classification.
// Outputs:
//   CLASSIFICATION_RESULT - ClassificationResult
//     The output aggregated classification results.
//
// The recommended way of using this subgraph is through the GraphBuilder API
// using the 'ConfigureClassificationPostprocessing()' function. See header file
// for more details.
class ClassificationPostprocessingSubgraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    ASSIGN_OR_RETURN(
        auto classification_result_out,
        BuildClassificationPostprocessing(
            sc->Options<ClassificationPostprocessingOptions>(),
            graph[Input<std::vector<Tensor>>(kTensorsTag)],
            graph[Input<std::vector<Timestamp>>(kTimestampsTag)], graph));
    classification_result_out >>
        graph[Output<ClassificationResult>(kClassificationResultTag)];
    return graph.GetConfig();
  }

 private:
  // Adds an on-device classification postprocessing subgraph into the provided
  // builder::Graph instance. The classification postprocessing subgraph takes
  // tensors (std::vector<mediapipe::Tensor>) as input and returns one output
  // stream containing the output classification results (ClassificationResult).
  //
  // options: the on-device ClassificationPostprocessingOptions.
  // tensors_in: (std::vector<mediapipe::Tensor>>) tensors to postprocess.
  // timestamps_in: (std::vector<mediapipe::Timestamp>) optional collection of
  //   timestamps that a single ClassificationResult should aggregate.
  // graph: the mediapipe builder::Graph instance to be updated.
  absl::StatusOr<Source<ClassificationResult>>
  BuildClassificationPostprocessing(
      const ClassificationPostprocessingOptions& options,
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
    GenericNode* tensors_dequantization_node;
    if (options.has_quantized_outputs()) {
      tensors_dequantization_node =
          &graph.AddNode("TensorsDequantizationCalculator");
      tensors_in >> tensors_dequantization_node->In(kTensorsTag);
    }

    // If there are multiple classification heads, the output tensors need to be
    // split.
    GenericNode* split_tensor_vector_node;
    if (num_heads > 1) {
      split_tensor_vector_node = &graph.AddNode("SplitTensorVectorCalculator");
      auto& split_tensor_vector_options =
          split_tensor_vector_node
              ->GetOptions<mediapipe::SplitVectorCalculatorOptions>();
      for (int i = 0; i < num_heads; ++i) {
        auto* range = split_tensor_vector_options.add_ranges();
        range->set_begin(i);
        range->set_end(i + 1);
      }
      if (options.has_quantized_outputs()) {
        tensors_dequantization_node->Out(kTensorsTag) >>
            split_tensor_vector_node->In(0);
      } else {
        tensors_in >> split_tensor_vector_node->In(0);
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
      if (num_heads == 1) {
        if (options.has_quantized_outputs()) {
          tensors_dequantization_node->Out(kTensorsTag) >>
              tensors_to_classification_nodes.back()->In(kTensorsTag);
        } else {
          tensors_in >> tensors_to_classification_nodes.back()->In(kTensorsTag);
        }
      } else {
        split_tensor_vector_node->Out(i) >>
            tensors_to_classification_nodes.back()->In(kTensorsTag);
      }
    }

    // Aggregates Classifications into a single ClassificationResult.
    auto& result_aggregation =
        graph.AddNode("ClassificationAggregationCalculator");
    result_aggregation.GetOptions<ClassificationAggregationCalculatorOptions>()
        .CopyFrom(options.classification_aggregation_options());
    for (int i = 0; i < num_heads; ++i) {
      tensors_to_classification_nodes[i]->Out(kClassificationsTag) >>
          result_aggregation.In(
              absl::StrFormat("%s:%d", kClassificationsTag, i));
    }
    timestamps_in >> result_aggregation.In(kTimestampsTag);

    // Connects output.
    return result_aggregation[Output<ClassificationResult>(
        kClassificationResultTag)];
  }
};
REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::ClassificationPostprocessingSubgraph);

}  // namespace tasks
}  // namespace mediapipe
