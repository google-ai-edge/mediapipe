// Copyright 2022 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_context.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::core::FindTensorIndexByMetadataName;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

constexpr absl::string_view kQueryTextMetadataName = "inp_text";
constexpr absl::string_view kResponseContextMetadataName = "res_context";
constexpr absl::string_view kResponseTextMetadataName = "res_text";

constexpr int kNumInputTensorsForUniversalSentenceEncoder = 3;

// Preprocesses input text into three kTfLiteString input tensors for a
// Universal Sentence Encoder (USE) model.
//
// The associated USE model is expected to contain input tensors with metadata
// names:
//
// Tensor           | Metadata Name
// ---------------- | ------------------
// Query text       | "inp_text"
// Response context | "res_context"
// Response text    | "res_text"
//
// This calculator will return an error if the model does not have three input
// tensors or if the tensors do not have metadata names corresponding to the
// above names in some order. Additional details regarding these input
// tensors are given in the Calculator "Outputs" section below.
//
// Inputs:
//   TEXT - std::string
//     The text to be embedded.
// Side Inputs:
//   METADATA_EXTRACTOR - ModelMetadataExtractor
//     The metadata extractor for the USE model. Used to determine the order of
//     the three input Tensors for the USE model.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing the three input Tensors for the USE model. The tensors
//     fit a question-answering setting and store a query text, a response
//     context, and a response text. This calculator will just be preprocessing
//     a single input text that will be stored in the response text tensor. The
//     query text and response context tensors will store empty strings.
//
// Example:
// node {
//   calculator: "UniversalSentenceEncoderPreprocessorCalculator"
//   input_stream: "TEXT:text"
//   input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
//   output_stream: "TENSORS:tensors"
// }
class UniversalSentenceEncoderPreprocessorCalculator : public Node {
 public:
  static constexpr Input<std::string> kTextIn{"TEXT"};
  static constexpr SideInput<ModelMetadataExtractor> kMetadataExtractorSideIn{
      "METADATA_EXTRACTOR"};
  static constexpr Output<std::vector<Tensor>> kTensorsOut{"TENSORS"};

  MEDIAPIPE_NODE_CONTRACT(kTextIn, kMetadataExtractorSideIn, kTensorsOut);

  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

  static absl::Status UpdateContract(CalculatorContract* cc);

 private:
  // Indices of the three input tensors for the USE model. They should form the
  // set {0, 1, 2}.
  int query_text_tensor_index_ = 0;
  int response_context_tensor_index_ = 1;
  int response_text_tensor_index_ = 2;

  // Tensor shapes for the model's input tensors.
  // The query text and response context tensors will only hold the empty
  // string, so their tensors will have shape [0], but the Universal Sentence
  // Encoder model's input signature requires them to be present. The response
  // text tensor will store the embedding text and have shape
  // [embedding_text_len].
  std::array<int, kNumInputTensorsForUniversalSentenceEncoder> tensor_shapes_;

  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status UniversalSentenceEncoderPreprocessorCalculator::Open(
    CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  const ModelMetadataExtractor* metadata_extractor =
      &kMetadataExtractorSideIn(cc).Get();
  auto* input_tensors_metadata = metadata_extractor->GetInputTensorMetadata();
  query_text_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kQueryTextMetadataName);
  response_context_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kResponseContextMetadataName);
  response_text_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kResponseTextMetadataName);

  absl::flat_hash_set<int> tensor_indices = absl::flat_hash_set<int>(
      {query_text_tensor_index_, response_context_tensor_index_,
       response_text_tensor_index_});
  if (tensor_indices != absl::flat_hash_set<int>({0, 1, 2})) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Input tensor indices form the set {$0, $1, $2} rather than {0, 1, 2}",
        query_text_tensor_index_, response_context_tensor_index_,
        response_text_tensor_index_));
  }
  return absl::OkStatus();
}

absl::Status UniversalSentenceEncoderPreprocessorCalculator::Process(
    CalculatorContext* cc) {
  absl::string_view text = kTextIn(cc).Get();
  const int text_len = static_cast<int>(text.length());
  tensor_shapes_[response_text_tensor_index_] = text_len;

  std::vector<Tensor> input_tensors;
  input_tensors.reserve(kNumInputTensorsForUniversalSentenceEncoder);
  for (int i = 0; i < kNumInputTensorsForUniversalSentenceEncoder; ++i) {
    input_tensors.push_back(
        {Tensor::ElementType::kChar,
         Tensor::Shape({tensor_shapes_[i]}, memory_manager_)});
  }

  std::memcpy(
      input_tensors[query_text_tensor_index_].GetCpuWriteView().buffer<char>(),
      "", 0);
  std::memcpy(input_tensors[response_context_tensor_index_]
                  .GetCpuWriteView()
                  .buffer<char>(),
              "", 0);
  std::memcpy(input_tensors[response_text_tensor_index_]
                  .GetCpuWriteView()
                  .buffer<char>(),
              text.data(), text_len * sizeof(char));
  kTensorsOut(cc).Send(std::move(input_tensors));
  return absl::OkStatus();
}

// static
absl::Status UniversalSentenceEncoderPreprocessorCalculator::UpdateContract(
    CalculatorContract* cc) {
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(UniversalSentenceEncoderPreprocessorCalculator);

}  // namespace api2
}  // namespace mediapipe
