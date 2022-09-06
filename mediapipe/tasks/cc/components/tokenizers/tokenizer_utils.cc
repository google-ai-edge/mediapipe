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

#include "mediapipe/tasks/cc/components/tokenizers/tokenizer_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/tokenizers/bert_tokenizer.h"
#include "mediapipe/tasks/cc/components/tokenizers/sentencepiece_tokenizer.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace tokenizer {

using ::mediapipe::tasks::CreateStatusWithPayload;
using ::mediapipe::tasks::MediaPipeTasksStatus;

namespace {

absl::StatusOr<absl::string_view> CheckAndLoadFirstAssociatedFile(
    const flatbuffers::Vector<flatbuffers::Offset<tflite::AssociatedFile>>*
        associated_files,
    const metadata::ModelMetadataExtractor* metadata_extractor) {
  if (associated_files == nullptr || associated_files->size() < 1 ||
      associated_files->Get(0)->name() == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid vocab_file from input process unit.",
        MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }
  ASSIGN_OR_RETURN(absl::string_view vocab_buffer,
                   metadata_extractor->GetAssociatedFile(
                       associated_files->Get(0)->name()->str()));
  return vocab_buffer;
}
}  // namespace

absl::StatusOr<std::unique_ptr<RegexTokenizer>> CreateRegexTokenizerFromOptions(
    const tflite::RegexTokenizerOptions* options,
    const metadata::ModelMetadataExtractor* metadata_extractor) {
  ASSIGN_OR_RETURN(absl::string_view vocab_buffer,
                   CheckAndLoadFirstAssociatedFile(options->vocab_file(),
                                                   metadata_extractor));
  if (options->delim_regex_pattern() == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "Invalid delim_regex_pattern from input process unit.",
        MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }

  std::unique_ptr<RegexTokenizer> regex_tokenizer =
      std::make_unique<RegexTokenizer>(options->delim_regex_pattern()->str(),
                                       vocab_buffer.data(),
                                       vocab_buffer.size());

  int unknown_token_id = 0;
  if (!regex_tokenizer->GetUnknownToken(&unknown_token_id)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "RegexTokenizer doesn't have <UNKNOWN> token.",
        MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }

  int pad_token_id = 0;
  if (!regex_tokenizer->GetPadToken(&pad_token_id)) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "RegexTokenizer doesn't have <PAD> token.",
        MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }

  return std::move(regex_tokenizer);
}

absl::StatusOr<std::unique_ptr<Tokenizer>> CreateTokenizerFromProcessUnit(
    const tflite::ProcessUnit* tokenizer_process_unit,
    const metadata::ModelMetadataExtractor* metadata_extractor) {
  if (metadata_extractor == nullptr || tokenizer_process_unit == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "No metadata or input process unit found.",
        MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }
  switch (tokenizer_process_unit->options_type()) {
    case tflite::ProcessUnitOptions_BertTokenizerOptions: {
      const tflite::BertTokenizerOptions* options =
          tokenizer_process_unit->options_as<tflite::BertTokenizerOptions>();
      ASSIGN_OR_RETURN(absl::string_view vocab_buffer,
                       CheckAndLoadFirstAssociatedFile(options->vocab_file(),
                                                       metadata_extractor));
      return std::make_unique<BertTokenizer>(vocab_buffer.data(),
                                             vocab_buffer.size());
    }
    case tflite::ProcessUnitOptions_SentencePieceTokenizerOptions: {
      const tflite::SentencePieceTokenizerOptions* options =
          tokenizer_process_unit
              ->options_as<tflite::SentencePieceTokenizerOptions>();
      ASSIGN_OR_RETURN(absl::string_view model_buffer,
                       CheckAndLoadFirstAssociatedFile(
                           options->sentencePiece_model(), metadata_extractor));
      return std::make_unique<SentencePieceTokenizer>(model_buffer.data(),
                                                      model_buffer.size());
    }
    case tflite::ProcessUnitOptions_RegexTokenizerOptions: {
      const tflite::RegexTokenizerOptions* options =
          tokenizer_process_unit->options_as<tflite::RegexTokenizerOptions>();
      return CreateRegexTokenizerFromOptions(options, metadata_extractor);
    }
    default:
      return CreateStatusWithPayload(
          absl::StatusCode::kNotFound,
          absl::StrCat("Incorrect options_type:",
                       tokenizer_process_unit->options_type()),
          MediaPipeTasksStatus::kMetadataInvalidTokenizerError);
  }
}

}  // namespace tokenizer
}  // namespace tasks
}  // namespace mediapipe
