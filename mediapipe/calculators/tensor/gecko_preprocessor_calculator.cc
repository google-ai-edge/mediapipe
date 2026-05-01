// Copyright 2026 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/gecko_preprocessor_calculator.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "mediapipe/calculators/tensor/gecko_preprocessor_calculator.pb.h"
#include "mediapipe/framework/api3/calculator.h"
#include "mediapipe/framework/api3/calculator_context.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/memory_manager_service.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/core/external_file_handler.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/cc/text/tokenizers/sentencepiece_tokenizer.h"
#include "mediapipe/tasks/cc/text/tokenizers/tokenizer.h"
#include "mediapipe/tasks/cc/text/tokenizers/tokenizer_utils.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {

using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::mediapipe::tasks::text::tokenizers::SentencePieceTokenizer;
using ::mediapipe::tasks::text::tokenizers::Tokenizer;

class GeckoPreprocessorCalculatorImpl
    : public api3::Calculator<GeckoPreprocessorCalculatorNode,
                              GeckoPreprocessorCalculatorImpl> {
 public:
  absl::Status Open(
      api3::CalculatorContext<GeckoPreprocessorCalculatorNode>& cc) override;
  absl::Status Process(
      api3::CalculatorContext<GeckoPreprocessorCalculatorNode>& cc) override;

 private:
  std::vector<int> GetPaddedTokens(const std::vector<int>& token_ids);
  std::unique_ptr<SentencePieceTokenizer> tokenizer_;
  std::unique_ptr<tasks::core::ExternalFileHandler>
      sentence_piece_model_handler_;
  // The max sequence length accepted by the text model.
  int max_seq_len_ = 0;
  // Enable pooling of AHWBs in Tensor instances.
  MemoryManager* memory_manager_ = nullptr;
};

absl::Status GeckoPreprocessorCalculatorImpl::Open(
    api3::CalculatorContext<GeckoPreprocessorCalculatorNode>& cc) {
  const auto& options = cc.options.Get();
  RET_CHECK(options.has_max_seq_len()) << "max_seq_len is required";
  RET_CHECK_GT(options.max_seq_len(), 0) << "max_seq_len must be positive";
  if (cc.Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc.Service(kMemoryManagerService).GetObject();
  }
  max_seq_len_ = options.max_seq_len();

  if (cc.Service(kMemoryManagerService).IsAvailable()) {
    memory_manager_ = &cc.Service(kMemoryManagerService).GetObject();
  }

  if (options.has_sentence_piece_model()) {
    MP_ASSIGN_OR_RETURN(
        sentence_piece_model_handler_,
        tasks::core::ExternalFileHandler::CreateFromExternalFile(
            &options.sentence_piece_model()));
    absl::string_view model_content =
        sentence_piece_model_handler_->GetFileContent();
    tokenizer_ = std::make_unique<SentencePieceTokenizer>(model_content.data(),
                                                          model_content.size());
    if (tokenizer_ == nullptr) {
      return absl::InternalError(
          "Failed to create SentencePieceTokenizer from ExternalFile.");
    }
    return absl::OkStatus();
  }

  const ModelMetadataExtractor* metadata_extractor =
      &cc.metadata_extractor.GetOrDie();
  const tflite::TensorMetadata* tensor_metadata =
      metadata_extractor->GetInputTensorMetadata(0);
  if (tensor_metadata == nullptr) {
    return absl::InvalidArgumentError("No tensor metadata found");
  }

  const tflite::ProcessUnit* tokenizer_process_unit = nullptr;
  for (const auto* process_unit : *tensor_metadata->process_units()) {
    if (process_unit->options_type() ==
        tflite::ProcessUnitOptions_SentencePieceTokenizerOptions) {
      tokenizer_process_unit = process_unit;
      break;
    }
  }
  if (tokenizer_process_unit == nullptr) {
    return absl::InvalidArgumentError(
        "No SentencePiece tokenizer process unit found in metadata.");
  }

  MP_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                      tasks::text::tokenizers::CreateTokenizerFromProcessUnit(
                          tokenizer_process_unit, metadata_extractor));
  auto* raw_tokenizer = dynamic_cast<SentencePieceTokenizer*>(tokenizer.get());
  if (raw_tokenizer == nullptr) {
    return absl::InternalError(
        "Failed to create SentencePieceTokenizer from metadata.");
  }
  tokenizer_.reset(dynamic_cast<SentencePieceTokenizer*>(tokenizer.release()));

  return absl::OkStatus();
}

std::vector<int> GeckoPreprocessorCalculatorImpl::GetPaddedTokens(
    const std::vector<int>& token_ids) {
  const size_t max_token_size = (token_ids.size() <= max_seq_len_ - 2)
                                    ? token_ids.size()
                                    : max_seq_len_ - 2;

  std::vector<int> padded_tokens(max_seq_len_, tokenizer_->pad_id());
  padded_tokens[0] = tokenizer_->bos_id();
  for (size_t i = 0; i < max_token_size; ++i) {
    padded_tokens[i + 1] = token_ids[i];
  }
  padded_tokens[max_token_size + 1] = tokenizer_->eos_id();
  return padded_tokens;
}

absl::Status GeckoPreprocessorCalculatorImpl::Process(
    api3::CalculatorContext<GeckoPreprocessorCalculatorNode>& cc) {
  std::vector<int> token_ids;
  tokenizer_->Encode(cc.text_in.GetOrDie(), &token_ids);

  RET_CHECK_LE(token_ids.size() + 2, max_seq_len_)
      << "Input text is too long: " << token_ids.size()
      << " tokens. size() + 2 <= max_seq_len_ (" << max_seq_len_
      << ") is required.";

  std::vector<int> input_tokens = GetPaddedTokens(token_ids);

  std::vector<Tensor> result;
  result.push_back({Tensor::ElementType::kInt32,
                    Tensor::Shape({1, max_seq_len_}), memory_manager_});
  std::memcpy(result[0].GetCpuWriteView().buffer<int32_t>(),
              input_tokens.data(), input_tokens.size() * sizeof(int32_t));
  cc.tensors_out.Send(std::move(result));
  return absl::OkStatus();
}

}  // namespace mediapipe
