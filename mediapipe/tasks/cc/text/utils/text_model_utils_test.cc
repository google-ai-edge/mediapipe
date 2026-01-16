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

#include <memory>
#include <string>

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/processors/proto/text_model_type.pb.h"
#include "mediapipe/tasks/cc/core/model_resources.h"
#include "mediapipe/tasks/cc/core/proto/external_file.pb.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe::tasks::text::utils {

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::components::processors::proto::TextModelType;
using ::mediapipe::tasks::core::ModelResources;
using ::mediapipe::tasks::core::proto::ExternalFile;

constexpr absl::string_view kTestModelResourcesTag = "test_model_resources";

constexpr absl::string_view kTestDataDirectory =
    "/mediapipe/tasks/testdata/text/";
// Classification model with BERT preprocessing.
constexpr absl::string_view kBertClassifierPath = "bert_text_classifier.tflite";
// Embedding model with BERT preprocessing.
constexpr absl::string_view kMobileBert =
    "mobilebert_embedding_with_metadata.tflite";
// Classification model with regex preprocessing.
constexpr absl::string_view kRegexClassifierPath =
    "test_model_text_classifier_with_regex_tokenizer.tflite";
// Embedding model with regex preprocessing.
constexpr absl::string_view kRegexOneEmbeddingModel =
    "regex_one_embedding_with_metadata.tflite";
// Classification model that takes a string tensor and outputs a bool tensor.
constexpr absl::string_view kStringToBoolModelPath =
    "test_model_text_classifier_bool_output.tflite";
constexpr char kUniversalSentenceEncoderModel[] =
    "universal_sentence_encoder_qa_with_metadata.tflite";

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

absl::StatusOr<TextModelType::ModelType> GetModelTypeFromFile(
    absl::string_view file_name) {
  auto model_file = std::make_unique<ExternalFile>();
  model_file->set_file_name(GetFullPath(file_name));
  MP_ASSIGN_OR_RETURN(
      auto model_resources,
      ModelResources::Create(std::string(kTestModelResourcesTag),
                             std::move(model_file)));
  return GetModelType(*model_resources);
}

}  // namespace

class TextModelUtilsTest : public tflite::testing::Test {};

TEST_F(TextModelUtilsTest, BertClassifierModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type,
                          GetModelTypeFromFile(kBertClassifierPath));
  ASSERT_EQ(model_type, TextModelType::BERT_MODEL);
}

TEST_F(TextModelUtilsTest, BertEmbedderModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type, GetModelTypeFromFile(kMobileBert));
  ASSERT_EQ(model_type, TextModelType::BERT_MODEL);
}

TEST_F(TextModelUtilsTest, RegexClassifierModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type,
                          GetModelTypeFromFile(kRegexClassifierPath));
  ASSERT_EQ(model_type, TextModelType::REGEX_MODEL);
}

TEST_F(TextModelUtilsTest, RegexEmbedderModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type,
                          GetModelTypeFromFile(kRegexOneEmbeddingModel));
  ASSERT_EQ(model_type, TextModelType::REGEX_MODEL);
}

TEST_F(TextModelUtilsTest, StringInputModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type,
                          GetModelTypeFromFile(kStringToBoolModelPath));
  ASSERT_EQ(model_type, TextModelType::STRING_MODEL);
}

TEST_F(TextModelUtilsTest, USEModelTest) {
  MP_ASSERT_OK_AND_ASSIGN(auto model_type,
                          GetModelTypeFromFile(kUniversalSentenceEncoderModel));
  ASSERT_EQ(model_type, TextModelType::USE_MODEL);
}

}  // namespace mediapipe::tasks::text::utils
