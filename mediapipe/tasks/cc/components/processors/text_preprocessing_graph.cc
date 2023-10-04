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
#include "mediapipe/tasks/cc/components/processors/text_preprocessing_graph.h"

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
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_model_type.pb.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_preprocessing_graph_options.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/text/utils/text_model_utils.h"

namespace mediapipe::tasks::components::processors {
namespace {

using ::mediapipe::api2::Input;
using ::mediapipe::api2::Output;
using ::mediapipe::api2::SideInput;
using ::mediapipe::api2::builder::Graph;
using ::mediapipe::api2::builder::SideSource;
using ::mediapipe::api2::builder::Source;
using ::mediapipe::tasks::components::processors::proto::TextModelType;
using ::mediapipe::tasks::components::processors::proto::
    TextPreprocessingGraphOptions;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::mediapipe::tasks::text::utils::GetModelType;

constexpr char kTextTag[] = "TEXT";
constexpr char kMetadataExtractorTag[] = "METADATA_EXTRACTOR";
constexpr char kTensorsTag[] = "TENSORS";

// Gets the name of the MediaPipe preprocessor calculator associated with
// `model_type`.
absl::StatusOr<std::string> GetCalculatorNameFromModelType(
    TextModelType::ModelType model_type) {
  switch (model_type) {
    case TextModelType::UNSPECIFIED_MODEL:
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument, "Unspecified model type",
          MediaPipeTasksStatus::kInvalidArgumentError);
    case TextModelType::BERT_MODEL:
      return "BertPreprocessorCalculator";
    case TextModelType::REGEX_MODEL:
      return "RegexPreprocessorCalculator";
    case TextModelType::STRING_MODEL:
      return "TextToTensorCalculator";
    case TextModelType::USE_MODEL:
      return "UniversalSentenceEncoderPreprocessorCalculator";
  }
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

// Determines whether the TFLite model for `model_graph` has input tensors with
// dynamic shape rather than static shape or returns an error if the input
// tensors have invalid shape signatures. This util assumes that the model has
// the correct input tensors type and count for the BertPreprocessorCalculator.
absl::StatusOr<bool> HasDynamicInputTensors(
    const tflite::SubGraph& model_graph) {
  const flatbuffers::Vector<int32_t>& input_indices = *model_graph.inputs();
  const flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>&
      model_tensors = *model_graph.tensors();

  // Static input tensors may have undefined shape signatures.
  if (absl::c_all_of(input_indices, [&model_tensors](int i) {
        return model_tensors[i]->shape_signature() == nullptr;
      })) {
    return false;
  } else if (absl::c_any_of(input_indices, [&model_tensors](int i) {
               return model_tensors[i]->shape_signature() == nullptr;
             })) {
    return CreateStatusWithPayload(absl::StatusCode::kInvalidArgument,
                                   "Input tensors contain a mix of defined and "
                                   "undefined shape signatures.");
  }

  for (int i : input_indices) {
    const tflite::Tensor* tensor = model_tensors[i];
    if (tensor->shape_signature()->size() != 2) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::Substitute(
              "Model should take 2-D shape signatures, got dimension: $0",
              tensor->shape_signature()->size()),
          MediaPipeTasksStatus::kInvalidInputTensorDimensionsError);
    }
  }

  // For dynamic input tensors, the shape_signature entry corresponding to the
  // input size is -1.
  if (absl::c_all_of(input_indices, [&model_tensors](int i) {
        return (*model_tensors[i]->shape_signature())[1] != -1;
      })) {
    return false;
  } else if (absl::c_all_of(input_indices, [&model_tensors](int i) {
               return (*model_tensors[i]->shape_signature())[1] == -1;
             })) {
    return true;
  } else {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Input tensors contain a mix of static and dynamic shapes.");
  }
  return false;
}

}  // namespace

absl::Status ConfigureTextPreprocessingGraph(
    const ModelResources& model_resources,
    TextPreprocessingGraphOptions& options) {
  if (model_resources.GetTfLiteModel()->subgraphs()->size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Text tflite models are assumed to have a single subgraph.",
        MediaPipeTasksStatus::kInvalidArgumentError);
  }

  MP_ASSIGN_OR_RETURN(TextModelType::ModelType model_type,
                      GetModelType(model_resources));
  const tflite::SubGraph& model_graph =
      *(*model_resources.GetTfLiteModel()->subgraphs())[0];
  options.set_model_type(model_type);
  switch (model_type) {
    case TextModelType::UNSPECIFIED_MODEL:
    case TextModelType::STRING_MODEL:
    case TextModelType::USE_MODEL: {
      break;
    }
    case TextModelType::BERT_MODEL:
    case TextModelType::REGEX_MODEL: {
      MP_ASSIGN_OR_RETURN(int max_seq_len, GetMaxSeqLen(model_graph));
      options.set_max_seq_len(max_seq_len);
    }
  }
  if (model_type == TextModelType::BERT_MODEL) {
    MP_ASSIGN_OR_RETURN(bool has_dynamic_input_tensors,
                        HasDynamicInputTensors(model_graph));
    options.set_has_dynamic_input_tensors(has_dynamic_input_tensors);
  }
  return absl::OkStatus();
}

// A TextPreprocessingGraph performs text preprocessing.
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
// using the 'ConfigureTextPreprocessingGraph()' function. See header file for
// more details.
class TextPreprocessingGraph : public mediapipe::Subgraph {
 public:
  absl::StatusOr<mediapipe::CalculatorGraphConfig> GetConfig(
      mediapipe::SubgraphContext* sc) override {
    Graph graph;
    MP_ASSIGN_OR_RETURN(
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
    MP_ASSIGN_OR_RETURN(std::string preprocessor_name,
                        GetCalculatorNameFromModelType(options.model_type()));
    auto& text_preprocessor = graph.AddNode(preprocessor_name);
    switch (options.model_type()) {
      case TextModelType::UNSPECIFIED_MODEL:
      case TextModelType::STRING_MODEL: {
        break;
      }
      case TextModelType::USE_MODEL: {
        metadata_extractor_in >>
            text_preprocessor.SideIn(kMetadataExtractorTag);
        break;
      }
      case TextModelType::BERT_MODEL: {
        text_preprocessor.GetOptions<BertPreprocessorCalculatorOptions>()
            .set_bert_max_seq_len(options.max_seq_len());
        text_preprocessor.GetOptions<BertPreprocessorCalculatorOptions>()
            .set_has_dynamic_input_tensors(options.has_dynamic_input_tensors());
        metadata_extractor_in >>
            text_preprocessor.SideIn(kMetadataExtractorTag);
        break;
      }
      case TextModelType::REGEX_MODEL: {
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
    ::mediapipe::tasks::components::processors::TextPreprocessingGraph);

}  // namespace mediapipe::tasks::components::processors
