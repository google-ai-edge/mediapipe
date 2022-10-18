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
#include "mediapipe/tasks/cc/components/text_preprocessing_graph.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/bert_preprocessor_calculator.pb.h"
#include "mediapipe/calculators/tensor/regex_preprocessor_calculator.pb.h"
#include "mediapipe/framework/api2/builder.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/subgraph.h"
#include "mediapipe/tasks/cc/components/proto/text_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {
namespace tasks {
namespace components {

namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SideInput;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SideSource;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::proto::TextPreprocessingGraphOptions;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

constexpr char kTextTag[] = "TEXT";
constexpr char kMetadataExtractorTag[] = "METADATA_EXTRACTOR";
constexpr char kTensorsTag[] = "TENSORS";

constexpr int kNumInputTensorsForBert = 3;
constexpr int kNumInputTensorsForRegex = 1;

// Gets the name of the MediaPipe calculator associated with
// `preprocessor_type`.
absl::StatusOr<std::string> GetCalculatorNameFromPreprocessorType(
    TextPreprocessingGraphOptions::PreprocessorType preprocessor_type) {
  switch (preprocessor_type) {
    case TextPreprocessingGraphOptions::UNSPECIFIED_PREPROCESSOR:
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument, "Unspecified preprocessor type",
          MediaPipeTasksStatus::kInvalidArgumentError);
    case TextPreprocessingGraphOptions::BERT_PREPROCESSOR:
      return "BertPreprocessorCalculator";
    case TextPreprocessingGraphOptions::REGEX_PREPROCESSOR:
      return "RegexPreprocessorCalculator";
    case TextPreprocessingGraphOptions::STRING_PREPROCESSOR:
      return "TextToTensorCalculator";
  }
}

// Determines the PreprocessorType for the model based on its metadata as well
// as its input tensors' type and count. Returns an error if there is no
// compatible preprocessor.
absl::StatusOr<TextPreprocessingGraphOptions::PreprocessorType>
GetPreprocessorType(const ModelResources& model_resources) {
  const tflite::SubGraph& model_graph =
      *(*model_resources.GetTfLiteModel()->subgraphs())[0];
  bool all_int32_tensors =
      absl::c_all_of(*model_graph.inputs(), [&model_graph](int i) {
        return (*model_graph.tensors())[i]->type() == tflite::TensorType_INT32;
      });
  bool all_string_tensors =
      absl::c_all_of(*model_graph.inputs(), [&model_graph](int i) {
        return (*model_graph.tensors())[i]->type() == tflite::TensorType_STRING;
      });
  if (!all_int32_tensors && !all_string_tensors) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "All input tensors should have type int32 or all should have type "
        "string",
        MediaPipeTasksStatus::kInvalidInputTensorTypeError);
  }
  if (all_string_tensors) {
    return TextPreprocessingGraphOptions::STRING_PREPROCESSOR;
  }

  // Otherwise, all tensors should have type int32
  const ModelMetadataExtractor* metadata_extractor =
      model_resources.GetMetadataExtractor();
  if (metadata_extractor->GetModelMetadata() == nullptr ||
      metadata_extractor->GetModelMetadata()->subgraph_metadata() == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Text models with int32 input tensors require TFLite Model "
        "Metadata but none was found",
        MediaPipeTasksStatus::kMetadataNotFoundError);
  }

  if (model_graph.inputs()->size() == kNumInputTensorsForBert) {
    return TextPreprocessingGraphOptions::BERT_PREPROCESSOR;
  }

  if (model_graph.inputs()->size() == kNumInputTensorsForRegex) {
    return TextPreprocessingGraphOptions::REGEX_PREPROCESSOR;
  }

  return CreateStatusWithPayload(
      absl::StatusCode::kInvalidArgument,
      absl::Substitute("Models with int32 input tensors should take exactly $0 "
                       "or $1 input tensors, but found $2",
                       kNumInputTensorsForBert, kNumInputTensorsForRegex,
                       model_graph.inputs()->size()),
      MediaPipeTasksStatus::kInvalidNumInputTensorsError);
}

// Returns the maximum input sequence length accepted by the TFLite
// model that owns `model graph` or returns an error if the model's input
// tensors' shape is invalid for text preprocessing. This util assumes that the
// model has the correct input tensors type and count for the
// BertPreprocessorCalculator or the RegexPreprocessorCalculator.
absl::StatusOr<int> GetMaxSeqLen(const tflite::SubGraph& model_graph) {
  const flatbuffers::Vector<int32_t>& input_indices = *model_graph.inputs();
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>&
      model_tensors = *model_graph.tensors();
  for (int i : input_indices) {
    const tflite::Tensor* tensor = model_tensors[i];

    if (tensor->shape()->size() != 2) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute(
              "Model should take 2-D input tensors, got dimension: $0",
              tensor->shape()->size()),
          MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
    }

    if ((*tensor->shape())[0] != 1) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute(
              "Input tensors should all have batch size 1, got: $0",
              (*tensor->shape())[0]),
          MediaPipeTasksStatus::kInvalidInputTensorSizeError);
    }
  }

  int max_seq_len = (*model_tensors[input_indices[0]]->shape())[1];
  if (!absl::c_all_of(input_indices, [&model_tensors, max_seq_len](int i) {
        return (*model_tensors[i]->shape())[1] == max_seq_len;
      })) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Input tensors don't have the same size",
        MediaPipeTasksStatus::kInvalidInputTensorSizeError);
  }
  return max_seq_len;
}
}  // namespace

absl::Status ConfigureTextPreprocessingSubgraph(
    const ModelResources& model_resources,
    TextPreprocessingGraphOptions& options) {
  if (model_resources.GetTfLiteModel()->subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Text tflite models are assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }

  ASSIGN_OR_RETURN(
      TextPreprocessingGraphOptions::PreprocessorType preprocessor_type,
      GetPreprocessorType(model_resources));
  options.set_preprocessor_type(preprocessor_type);
  switch (preprocessor_type) {
    case TextPreprocessingGraphOptions::UNSPECIFIED_PREPROCESSOR:
    case TextPreprocessingGraphOptions::STRING_PREPROCESSOR: {
      break;
    }
    case TextPreprocessingGraphOptions::BERT_PREPROCESSOR:
    case TextPreprocessingGraphOptions::REGEX_PREPROCESSOR: {
      ASSIGN_OR_RETURN(
          int max_seq_len,
          GetMaxSeqLen(*(*model_resources.GetTfLiteModel()->subgraphs())[0]));
      options.set_max_seq_len(max_seq_len);
    }
  }

  return absl::OkStatus();
}

// A "mediapipe.tasks.components.TextPreprocessingSubgraph" performs text
// preprocessing.
// - Accepts a std::string input and outputs CPU tensors.
//
// Inputs:
//   TEXT - std::string
//     The text to preprocess.
// Side inputs:
//   METADATA_EXTRACTOR - ModelMetadataExtractor
//     The metadata extractor for the TFLite model. Used to determine the order
//     for input tensors and to extract tokenizer information.
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing the preprocessed input tensors for the TFLite model.
//
// The recommended way of using this subgraph is through the GraphBuilder API
// using the 'ConfigureTextPreprocessing()' function. See header file for more
// details.
class TextPreprocessingSubgraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    ASSIGN_OR_RETURN(
        Source<std::vector<Tensor>> tensors_in,
        BuildTextPreprocessing(
            sc->Options<TextPreprocessingGraphOptions>(),
            graph[Input<std::string>(kTextTag)],
            graph[SideInput<ModelMetadataExtractor>(kMetadataExtractorTag)],
            graph));
    tensors_in >> graph[Output<std::vector<Tensor>>(kTensorsTag)];
    return graph.GetConfig();
  }

 private:
  absl::StatusOr<Source<std::vector<Tensor>>> BuildTextPreprocessing(
      const TextPreprocessingGraphOptions& options, Source<std::string> text_in,
      SideSource<ModelMetadataExtractor> metadata_extractor_in, Graph& graph) {
    ASSIGN_OR_RETURN(
        std::string preprocessor_name,
        GetCalculatorNameFromPreprocessorType(options.preprocessor_type()));
    auto& text_preprocessor = graph.AddNode(preprocessor_name);
    switch (options.preprocessor_type()) {
      case TextPreprocessingGraphOptions::UNSPECIFIED_PREPROCESSOR:
      case TextPreprocessingGraphOptions::STRING_PREPROCESSOR: {
        break;
      }
      case TextPreprocessingGraphOptions::BERT_PREPROCESSOR: {
        text_preprocessor.GetOptions<BertPreprocessorCalculatorOptions>()
            .set_bert_max_seq_len(options.max_seq_len());
        metadata_extractor_in >>
            text_preprocessor.SideIn(kMetadataExtractorTag);
        break;
      }
      case TextPreprocessingGraphOptions::REGEX_PREPROCESSOR: {
        text_preprocessor.GetOptions<RegexPreprocessorCalculatorOptions>()
            .set_max_seq_len(options.max_seq_len());
        metadata_extractor_in >>
            text_preprocessor.SideIn(kMetadataExtractorTag);
        break;
      }
    }
    text_in >> text_preprocessor.In(kTextTag);
    return text_preprocessor[Output<std::vector<Tensor>>(kTensorsTag)];
  }
};
REGISTER_MEDIAPIPE_GRAPH(
    ::mediapipe::tasks::components::TextPreprocessingSubgraph);

}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
