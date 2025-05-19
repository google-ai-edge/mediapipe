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

#include "mediapipe/tasks/cc/text/utils/text_model_utils.h"

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_model_type.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace mediapipe::tasks::text::utils {
namespace {

using ::mediapipe::tasks::components::processors::proto::TextModelType;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

constexpr int kNumInputTensorsForBert = 3;
constexpr int kNumInputTensorsForRegex = 1;
constexpr int kNumInputTensorsForStringPreprocessor = 1;
constexpr int kNumInputTensorsForUSE = 3;

// Determines the ModelType for a model with int32 input tensors based
// on the number of input tensors. Returns an error if there is missing metadata
// or an invalid number of input tensors.
absl::StatusOr<TextModelType::ModelType> GetIntTensorModelType(
    const ModelResources& model_resources, int num_input_tensors) {
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

  if (num_input_tensors == kNumInputTensorsForBert) {
    return TextModelType::BERT_MODEL;
  }

  if (num_input_tensors == kNumInputTensorsForRegex) {
    return TextModelType::REGEX_MODEL;
  }

  return CreateStatusWithPayload(
      absl::StatusCode::kInvalidArgument,
      absl::Substitute("Models with int32 input tensors should take exactly $0 "
                       "or $1 input tensors, but found $2",
                       kNumInputTensorsForBert, kNumInputTensorsForRegex,
                       num_input_tensors),
      MediaPipeTasksStatus::kInvalidNumInputTensorsError);
}

// Determines the ModelType for a model with string input tensors based
// on the number of input tensors. Returns an error if there is an invalid
// number of input tensors.
absl::StatusOr<TextModelType::ModelType> GetStringTensorModelType(
    const ModelResources& model_resources, int num_input_tensors) {
  if (num_input_tensors == kNumInputTensorsForStringPreprocessor) {
    return TextModelType::STRING_MODEL;
  }

  if (num_input_tensors == kNumInputTensorsForUSE) {
    return TextModelType::USE_MODEL;
  }

  return CreateStatusWithPayload(
      absl::StatusCode::kInvalidArgument,
      absl::Substitute("Models with string input tensors should take exactly "
                       "$0 or $1 input tensors, but found $2",
                       kNumInputTensorsForStringPreprocessor,
                       kNumInputTensorsForUSE, num_input_tensors),
      MediaPipeTasksStatus::kInvalidNumInputTensorsError);
}
}  // namespace

absl::StatusOr<TextModelType::ModelType> GetModelType(
    const ModelResources& model_resources) {
  const tflite::SubGraph& model_graph =
      *(*model_resources.GetTfLiteModel()->subgraphs())[0];
  bool all_int32_tensors =
      absl::c_all_of(*model_graph.inputs(), [&model_graph](int i) {
        return (*model_graph.tensors())[i] -> type() ==
                                                  tflite::TensorType_INT32;
      });
  bool all_string_tensors =
      absl::c_all_of(*model_graph.inputs(), [&model_graph](int i) {
        return (*model_graph.tensors())[i] -> type() ==
                                                  tflite::TensorType_STRING;
      });
  if (!all_int32_tensors && !all_string_tensors) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "All input tensors should have type int32 or all should have type "
        "string",
        MediaPipeTasksStatus::kInvalidInputTensorTypeError);
  }
  if (all_string_tensors) {
    return GetStringTensorModelType(model_resources,
                                    model_graph.inputs()->size());
  }

  // Otherwise, all tensors should have type int32
  return GetIntTensorModelType(model_resources, model_graph.inputs()->size());
}

}  // namespace mediapipe::tasks::text::utils
