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

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "mediapipe/calculators/tensor/regex_preprocessor_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/api2/port.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/text/tokenizers/regex_tokenizer.h"
#include "mediapipe/tasks/cc/text/tokenizers/tokenizer_utils.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace api2 {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;

// Preprocesses input text into one int32 input tensor for a text model using
// a RegexTokenizer.
//
// Inputs:
//   TEXT - std::string
//     The input text.
// Side Inputs:
//   METADATA_EXTRACTOR - ModelMetadataExtractor
//     The metadata extractor for the text model. Used to extract the metadata
//     to construct the RegexTokenizer.
//
// Outputs:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor which is the text model's input tensor.
//     Depending on the tokenizer metadata, the tensor may start with
//     the id of the tokenizer's <START> token. The following tensor values will
//     be the ids of the tokens of the input text. Any out-of-vocab tokens will
//     have the id of the <UNKNOWN> token. The tensor will be padded with the
//     <PAD> token id to have size equal to the max sequence length for the text
//     model.
//
// Example:
// node {
//   calculator: "RegexPreprocessorCalculator"
//   input_stream: "TEXT:text"
//   input_side_packet: "METADATA_EXTRACTOR:metadata_extractor"
//   output_stream: "TENSORS:tensors"
//   options {
//     [mediapipe.RegexPreprocessorCalculatorOptions.ext] {
//       max_seq_len: 256
//     }
//   }
// }
class RegexPreprocessorCalculator : public Node {
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
  std::unique_ptr<tasks::text::tokenizers::RegexTokenizer> tokenizer_;
  // The max sequence length accepted by the text model.
  int max_seq_len_ = 0;
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status RegexPreprocessorCalculator::UpdateContract(
    CalculatorContract* cc) {
  const auto& options =
      cc->Options<mediapipe::RegexPreprocessorCalculatorOptions>();
  RET_CHECK(options.has_max_seq_len()) << "max_seq_len is required";
  RET_CHECK_GT(options.max_seq_len(), 0) << "max_seq_len must be positive";
  cc->UseService(kMemoryManagerService).Optional();
  return absl::OkStatus();
}

absl::Status RegexPreprocessorCalculator::Open(CalculatorContext* cc) {
  if (cc->Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc->Service(kMemoryManagerService).GetObject();
  }
  const ModelMetadataExtractor* metadata_extractor =
      &kMetadataExtractorSideIn(cc).Get();
  const tflite::TensorMetadata* tensor_metadata =
      metadata_extractor->GetInputTensorMetadata(0);
  if (tensor_metadata == nullptr) {
    return absl::InvalidArgumentError("No tensor metadata found");
  }

  MP_ASSIGN_OR_RETURN(
      const auto* tokenizer_metadata,
      metadata_extractor->FindFirstProcessUnit(
          *tensor_metadata, tflite::ProcessUnitOptions_RegexTokenizerOptions));
  if (tokenizer_metadata == nullptr) {
    return absl::InvalidArgumentError("No tokenizer metadata found");
  }
  const tflite::RegexTokenizerOptions* regex_tokenizer_options =
      tokenizer_metadata->options_as<tflite::RegexTokenizerOptions>();
  MP_ASSIGN_OR_RETURN(tokenizer_,
                      tasks::text::tokenizers::CreateRegexTokenizerFromOptions(
                          regex_tokenizer_options, metadata_extractor));

  const auto& options =
      cc->Options<mediapipe::RegexPreprocessorCalculatorOptions>();
  max_seq_len_ = options.max_seq_len();
  return absl::OkStatus();
}

absl::Status RegexPreprocessorCalculator::Process(CalculatorContext* cc) {
  tasks::text::tokenizers::TokenizerResult tokenizer_result =
      tokenizer_->Tokenize(kTextIn(cc).Get());

  int unknown_token_id = 0;
  tokenizer_->GetUnknownToken(&unknown_token_id);
  int pad_token_id = 0;
  tokenizer_->GetPadToken(&pad_token_id);

  std::vector<int> input_tokens(max_seq_len_, pad_token_id);
  int start_token_id = 0;
  int input_token_index = 0;
  if (tokenizer_->GetStartToken(&start_token_id)) {
    input_tokens[0] = start_token_id;
    input_token_index = 1;
  }

  for (int i = 0; (i < tokenizer_result.subwords.size()) &&
                  (input_token_index < max_seq_len_);
       ++i, ++input_token_index) {
    const std::string& token = tokenizer_result.subwords[i];
    int token_id = 0;
    if (tokenizer_->LookupId(token, &token_id)) {
      input_tokens[input_token_index] = token_id;
    } else {
      input_tokens[input_token_index] = unknown_token_id;
    }
  }

  //                              |<-------sentence_length-------->|
  // input_tensor                 <START>, t1, t2... <PAD>, <PAD>...
  // <START> is optional, t1, t2... will be replaced by <UNKNOWN> if it's
  // not found in the tokenizer vocab.
  std::vector<Tensor> result;
  result.push_back({Tensor::ElementType::kInt32,
                    Tensor::Shape({1, max_seq_len_}), memory_manager_});
  std::memcpy(result[0].GetCpuWriteView().buffer<int32_t>(),
              input_tokens.data(), input_tokens.size() * sizeof(int32_t));
  kTensorsOut(cc).Send(std::move(result));
  return absl::OkStatus();
}

MEDIAPIPE_REGISTER_NODE(RegexPreprocessorCalculator);

}  // namespace api2
}  // namespace mediapipe
