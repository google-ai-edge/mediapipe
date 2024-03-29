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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/calculators/tensor/bert_preprocessor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/text/tokenizers/tokenizer.h"
#include "mediapipe/tasks/cc/text/tokenizers/tokenizer_utils.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::core::FindTensorIndexByMetadataName;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

constexpr int kNumInputTensorsForBert = 3;
constexpr int kTokenizerProcessUnitIndex = 0;
constexpr absl::string_view kInputIdsTensorName = "ids";
constexpr absl::string_view kInputMasksTensorName = "mask";
constexpr absl::string_view kSegmentIdsTensorName = "segment_ids";
constexpr absl::string_view kClassifierToken = "[CLS]";
constexpr absl::string_view kSeparatorToken = "[SEP]";

// Preprocesses input text into three int32 input tensors for a BERT model using
// a tokenizer.
// The associated BERT model is expected to contain input tensors with names:
//
// Tensor           | Metadata Name
// ---------------- | --------------
// IDs              | "ids"
// Segment IDs      | "segment_ids"
// Mask             | "mask"
//
// This calculator will return an error if the model does not have three input
// tensors or if the tensors do not have names corresponding to the above
// metadata names in some order. Additional details regarding these input
// tensors are given in the Calculator "Outputs" section below.
//
// This calculator is currently configured for the TextClassifier Task but it
// will eventually be generalized for other Text Tasks.
//
// Inputs:
//   TEXT - std::string
//     The input text.
// Side Inputs:
//   METADATA_EXTRACTOR - ModelMetadataExtractor
//     The metadata extractor for the BERT model. Used to determine the order of
//     the three input Tensors for the BERT model and to extract the metadata to
//     construct the tokenizer.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing the three input Tensors for the BERT model:
//       (1): the token ids of the tokenized input string. A classifier token
//            ("[CLS]") will be prepended to the input tokens and a separator
//            token ("[SEP]") will be appended to the input tokens.
//       (2): the segment ids, which are all 0 for now but will have different
//            values to distinguish between different sentences in the input
//            text for other Text tasks.
//       (3): the input mask ids, which are 1 at each of the input token indices
//            and 0 elsewhere.
//     The Tensors will have size equal to the max sequence length for the BERT
//     model.
//
// Example:
// node {
//   calculator: "BertPreprocessorCalculator"
//   input_stream: "TEXT:text"
//   input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
//   output_stream: "TENSORS:tensors"
//   options {
//     [mediapipe.BertPreprocessorCalculatorOptions.ext] {
//       bert_max_seq_len: 128
//     }
//   }
// }
class BertPreprocessorCalculator : public Node {
 public:
  static constexpr Input<std::string> kTextIn{"TEXT"};
  static constexpr SideInput<ModelMetadataExtractor> kMetadataExtractorSideIn{
      "METADATA_EXTRACTOR"};
  static constexpr Output<std::vector<Tensor>> kTensorsOut{"TENSORS"};

  MEDIAPIPE_NODE_CONTRACT(kTextIn, kMetadataExtractorSideIn, kTensorsOut);

  static absl::Status UpdateContract(CalculatorContract* cc);
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;

 private:
  std::unique_ptr<tasks::text::tokenizers::Tokenizer> tokenizer_;
  // The max sequence length accepted by the BERT model if its input tensors
  // are static.
  int bert_max_seq_len_ = 2;
  // Indices of the three input tensors for the BERT model. They should form the
  // set {0, 1, 2}.
  int input_ids_tensor_index_ = 0;
  int segment_ids_tensor_index_ = 1;
  int input_masks_tensor_index_ = 2;
  // Whether the model's input tensor shapes are dynamic.
  bool has_dynamic_input_tensors_ = false;

  // Applies `tokenizer_` to the `input_text` to generate a vector of tokens.
  // This util prepends "[CLS]" and appends "[SEP]" to the input tokens and
  // clips the vector of tokens to have length at most `bert_max_seq_len_` if
  // the input tensors are static.
  std::vector<std::string> TokenizeInputText(absl::string_view input_text);
  // Processes the `input_tokens` to generate the three input tensors of size
  // `tensor_size` for the BERT model.
  std::vector<Tensor> GenerateInputTensors(
      const std::vector<std::string>& input_tokens, int tensor_size);

  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status BertPreprocessorCalculator::UpdateContract(
    CalculatorContract* cc) {
  const auto& options =
      cc->Options<mediapipe::BertPreprocessorCalculatorOptions>();
  if (options.has_dynamic_input_tensors()) {
    return absl::OkStatus();
  } else {
    RET_CHECK(options.has_bert_max_seq_len()) << "bert_max_seq_len is required";
    RET_CHECK_GE(options.bert_max_seq_len(), 2)
        << "bert_max_seq_len must be at least 2";
  }
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

absl::Status BertPreprocessorCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  const ModelMetadataExtractor* metadata_extractor =
      &kMetadataExtractorSideIn(cc).Get();
  const tflite::ProcessUnit* tokenizer_metadata =
      metadata_extractor->GetInputProcessUnit(kTokenizerProcessUnitIndex);
  MP_ASSIGN_OR_RETURN(tokenizer_,
                      tasks::text::tokenizers::CreateTokenizerFromProcessUnit(
                          tokenizer_metadata, metadata_extractor));

  auto* input_tensors_metadata = metadata_extractor->GetInputTensorMetadata();
  input_ids_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kInputIdsTensorName);
  segment_ids_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kSegmentIdsTensorName);
  input_masks_tensor_index_ = FindTensorIndexByMetadataName(
      input_tensors_metadata, kInputMasksTensorName);
  absl::flat_hash_set<int> tensor_indices = {input_ids_tensor_index_,
                                             segment_ids_tensor_index_,
                                             input_masks_tensor_index_};
  if (tensor_indices != absl::flat_hash_set<int>({0, 1, 2})) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Input tensor indices form the set {$0, $1, $2} rather than {0, 1, 2}",
        input_ids_tensor_index_, segment_ids_tensor_index_,
        input_masks_tensor_index_));
  }

  const auto& options =
      cc->Options<mediapipe::BertPreprocessorCalculatorOptions>();
  bert_max_seq_len_ = options.bert_max_seq_len();
  has_dynamic_input_tensors_ = options.has_dynamic_input_tensors();
  return absl::OkStatus();
}

absl::Status BertPreprocessorCalculator::Process(CalculatorContext* cc) {
  int tensor_size = bert_max_seq_len_;
  std::vector<std::string> input_tokens = TokenizeInputText(kTextIn(cc).Get());
  if (has_dynamic_input_tensors_) {
    tensor_size = input_tokens.size();
  }
  kTensorsOut(cc).Send(GenerateInputTensors(input_tokens, tensor_size));
  return absl::OkStatus();
}

std::vector<std::string> BertPreprocessorCalculator::TokenizeInputText(
    absl::string_view input_text) {
  std::string processed_input = std::string(input_text);
  absl::AsciiStrToLower(&processed_input);

  tasks::text::tokenizers::TokenizerResult tokenizer_result =
      tokenizer_->Tokenize(processed_input);

  // Offset by 2 to account for [CLS] and [SEP]
  int input_tokens_size =
      static_cast<int>(tokenizer_result.subwords.size()) + 2;
  // For static shapes, truncate the input tokens to `bert_max_seq_len_`.
  if (!has_dynamic_input_tensors_) {
    input_tokens_size = std::min(bert_max_seq_len_, input_tokens_size);
  }
  std::vector<std::string> input_tokens;
  input_tokens.reserve(input_tokens_size);
  input_tokens.push_back(std::string(kClassifierToken));
  for (int i = 0; i < input_tokens_size - 2; ++i) {
    input_tokens.push_back(std::move(tokenizer_result.subwords[i]));
  }
  input_tokens.push_back(std::string(kSeparatorToken));
  return input_tokens;
}

std::vector<Tensor> BertPreprocessorCalculator::GenerateInputTensors(
    const std::vector<std::string>& input_tokens, int tensor_size) {
  std::vector<int32_t> input_ids(tensor_size, 0);
  std::vector<int32_t> segment_ids(tensor_size, 0);
  std::vector<int32_t> input_masks(tensor_size, 0);
  // Convert tokens back into ids and set mask
  for (int i = 0; i < input_tokens.size(); ++i) {
    tokenizer_->LookupId(input_tokens[i], &input_ids[i]);
    input_masks[i] = 1;
  }
  //                           |<-----------tensor_size------------>|
  // input_ids                 [CLS] s1  s2...  sn [SEP]  0  0...  0
  // segment_ids                 0    0   0...  0    0    0  0...  0
  // input_masks                 1    1   1...  1    1    0  0...  0

  std::vector<Tensor> input_tensors;
  input_tensors.reserve(kNumInputTensorsForBert);
  for (int i = 0; i < kNumInputTensorsForBert; ++i) {
    input_tensors.push_back(
        {Tensor::ElementType::kInt32,
         Tensor::Shape({1, tensor_size}, has_dynamic_input_tensors_),
         memory_manager_});
  }
  std::memcpy(input_tensors[input_ids_tensor_index_]
                  .GetCpuWriteView()
                  .buffer<int32_t>(),
              input_ids.data(), input_ids.size() * sizeof(int32_t));
  std::memcpy(input_tensors[segment_ids_tensor_index_]
                  .GetCpuWriteView()
                  .buffer<int32_t>(),
              segment_ids.data(), segment_ids.size() * sizeof(int32_t));
  std::memcpy(input_tensors[input_masks_tensor_index_]
                  .GetCpuWriteView()
                  .buffer<int32_t>(),
              input_masks.data(), input_masks.size() * sizeof(int32_t));
  return input_tensors;
}

MEDIAPIPE_REGISTER_NODE(BertPreprocessorCalculator);

}  // namespace api2
}  // namespace mediapipe
