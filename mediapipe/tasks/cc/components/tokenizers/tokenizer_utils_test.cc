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
#include <type_traits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "flatbuffers/flatbuffers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/tokenizers/bert_tokenizer.h"
#include "mediapipe/tasks/cc/components/tokenizers/regex_tokenizer.h"
#include "mediapipe/tasks/cc/components/tokenizers/sentencepiece_tokenizer.h"
#include "mediapipe/tasks/cc/core/utils.h"
#include "mediapipe/tasks/cc/metadata/metadata_extractor.h"
#include "mediapipe/tasks/metadata/metadata_schema_generated.h"

namespace mediapipe {
namespace tasks {
namespace tokenizer {

using ::mediapipe::tasks::kMediaPipeTasksPayload;
using ::mediapipe::tasks::MediaPipeTasksStatus;
using ::mediapipe::tasks::core::LoadBinaryContent;
using ::mediapipe::tasks::metadata::ModelMetadataExtractor;
using ::testing::HasSubstr;

namespace {
constexpr char kModelWithBertTokenizerPath[] =
    "mediapipe/tasks/testdata/text/"
    "mobilebert_with_metadata.tflite";
constexpr char kModelWithSentencePieceTokenizerPath[] =
    "mediapipe/tasks/testdata/text/"
    "albert_with_metadata.tflite";
constexpr char kModelWithRegexTokenizerPath[] =
    "mediapipe/tasks/testdata/text/"
    "test_model_nl_classifier_with_regex_tokenizer.tflite";

template <typename TargetType, typename T>
bool is_type(T* t) {
  return dynamic_cast<TargetType*>(t) != nullptr;
}

}  // namespace

TEST(TokenizerUtilsTest, TestCreateMobileBertTokenizer) {
  std::string model_buffer = LoadBinaryContent(kModelWithBertTokenizerPath);

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> metadata_extractor,
      ModelMetadataExtractor::CreateFromModelBuffer(model_buffer.data(),
                                                    model_buffer.size()));
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Tokenizer> tokenizer,
      CreateTokenizerFromProcessUnit(metadata_extractor->GetInputProcessUnit(0),
                                     metadata_extractor.get()));
  ASSERT_TRUE(is_type<BertTokenizer>(tokenizer.get()));
}

TEST(TokenizerUtilsTest, TestCreateAlBertTokenizer) {
  std::string model_buffer =
      LoadBinaryContent(kModelWithSentencePieceTokenizerPath);

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> metadata_extractor,
      ModelMetadataExtractor::CreateFromModelBuffer(model_buffer.data(),
                                                    model_buffer.size()));
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Tokenizer> tokenizer,
      CreateTokenizerFromProcessUnit(metadata_extractor->GetInputProcessUnit(0),
                                     metadata_extractor.get()));
  ASSERT_TRUE(is_type<SentencePieceTokenizer>(tokenizer.get()));
}

TEST(TokenizerUtilsTest, TestCreateRegexTokenizer) {
  std::string model_buffer = LoadBinaryContent(kModelWithRegexTokenizerPath);

  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModelMetadataExtractor> metadata_extractor,
      ModelMetadataExtractor::CreateFromModelBuffer(model_buffer.data(),
                                                    model_buffer.size()));
  MP_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Tokenizer> tokenizer,
      CreateTokenizerFromProcessUnit(
          metadata_extractor->GetInputTensorMetadata(0)->process_units()->Get(
              0),
          metadata_extractor.get()));
  ASSERT_TRUE(is_type<RegexTokenizer>(tokenizer.get()));
}

TEST(TokenizerUtilsTest, TestCreateFailure) {
  absl::StatusOr<std::unique_ptr<Tokenizer>> tokenizer_status =
      CreateTokenizerFromProcessUnit(nullptr, nullptr);

  EXPECT_THAT(tokenizer_status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No metadata or input process unit found.")));
  EXPECT_THAT(tokenizer_status.status().GetPayload(kMediaPipeTasksPayload),
              testing::Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kMetadataInvalidTokenizerError))));
}

}  // namespace tokenizer
}  // namespace tasks
}  // namespace mediapipe
