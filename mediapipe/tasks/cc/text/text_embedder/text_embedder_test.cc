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

#include "mediapipe/tasks/cc/text/text_embedder/text_embedder.h"

#include <memory>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe::tasks::text::text_embedder {
namespace {

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";

// Note that these models use dynamic-sized tensors.
// Embedding model with BERT preprocessing.
constexpr char kMobileBert[] = "mobilebert_embedding_with_metadata.tflite";
// Embedding model with regex preprocessing.
constexpr char kRegexOneEmbeddingModel[] =
    "regex_one_embedding_with_metadata.tflite";
constexpr char kUniversalSentenceEncoderModel[] =
    "universal_sentence_encoder_qa_with_metadata.tflite";

// Tolerance for embedding vector coordinate values.
constexpr float kEpsilon = 1e-4;
// Tolerance for cosine similarity evaluation.
constexpr double kSimilarityTolerancy = 2e-2;

using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

class EmbedderTest : public tflite::testing::Test {};

TEST_F(EmbedderTest, FailsWithMissingModel) {
  auto text_embedder =
      TextEmbedder::Create(std::make_unique<TextEmbedderOptions>());
  ASSERT_EQ(text_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  ASSERT_THAT(
      text_embedder.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  ASSERT_THAT(text_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(EmbedderTest, SucceedsWithMobileBert) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileBert);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result0,
      text_embedder->Embed("it's a charming and often affecting journey"));
  ASSERT_EQ(result0.embeddings.size(), 1);
  ASSERT_EQ(result0.embeddings[0].float_embedding.size(), 512);
#ifdef _WIN32
  ASSERT_NEAR(result0.embeddings[0].float_embedding[0], 21.2148f, kEpsilon);
#elif defined(__FMA__)
  ASSERT_NEAR(result0.embeddings[0].float_embedding[0], 21.3605f, kEpsilon);
#else
  ASSERT_NEAR(result0.embeddings[0].float_embedding[0], 21.2054f, kEpsilon);
#endif  // _WIN32

  MP_ASSERT_OK_AND_ASSIGN(
      auto result1, text_embedder->Embed("what a great and fantastic trip"));
  ASSERT_EQ(result1.embeddings.size(), 1);
  ASSERT_EQ(result1.embeddings[0].float_embedding.size(), 512);
#ifdef __FMA__
  ASSERT_NEAR(result1.embeddings[0].float_embedding[0], 21.254150f, kEpsilon);
#else
  ASSERT_NEAR(result1.embeddings[0].float_embedding[0], 22.387123f, kEpsilon);
#endif

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
#ifdef _WIN32
  EXPECT_NEAR(similarity, 0.971417, kSimilarityTolerancy);
#else
  EXPECT_NEAR(similarity, 0.969514, kSimilarityTolerancy);
#endif  // _WIN32

  MP_ASSERT_OK(text_embedder->Close());
}

TEST(EmbedTest, SucceedsWithRegexOneEmbeddingModel) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kRegexOneEmbeddingModel);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(
      auto result0,
      text_embedder->Embed("it's a charming and often affecting journey"));
  EXPECT_EQ(result0.embeddings.size(), 1);
  EXPECT_EQ(result0.embeddings[0].float_embedding.size(), 16);

  EXPECT_NEAR(result0.embeddings[0].float_embedding[0], 0.0309356f, kEpsilon);

  MP_ASSERT_OK_AND_ASSIGN(
      auto result1, text_embedder->Embed("what a great and fantastic trip"));
  EXPECT_EQ(result1.embeddings.size(), 1);
  EXPECT_EQ(result1.embeddings[0].float_embedding.size(), 16);

  EXPECT_NEAR(result1.embeddings[0].float_embedding[0], 0.0312863f, kEpsilon);

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  EXPECT_NEAR(similarity, 0.999937, kSimilarityTolerancy);

  MP_ASSERT_OK(text_embedder->Close());
}

TEST_F(EmbedderTest, SucceedsWithQuantization) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileBert);
  options->embedder_options.quantize = true;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result,
      text_embedder->Embed("it's a charming and often affecting journey"));
  ASSERT_EQ(result.embeddings.size(), 1);
  ASSERT_EQ(result.embeddings[0].quantized_embedding.size(), 512);

  MP_ASSERT_OK(text_embedder->Close());
}

TEST(EmbedTest, SucceedsWithUniversalSentenceEncoderModel) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kUniversalSentenceEncoderModel);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(
      auto result0,
      text_embedder->Embed("it's a charming and often affecting journey"));
  ASSERT_EQ(result0.embeddings.size(), 1);
  ASSERT_EQ(result0.embeddings[0].float_embedding.size(), 100);
  ASSERT_NEAR(result0.embeddings[0].float_embedding[0], 1.422951f, kEpsilon);

  MP_ASSERT_OK_AND_ASSIGN(
      auto result1, text_embedder->Embed("what a great and fantastic trip"));
  ASSERT_EQ(result1.embeddings.size(), 1);
  ASSERT_EQ(result1.embeddings[0].float_embedding.size(), 100);
  ASSERT_NEAR(result1.embeddings[0].float_embedding[0], 1.404664f, kEpsilon);

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  ASSERT_NEAR(similarity, 0.851961, kSimilarityTolerancy);

  MP_ASSERT_OK(text_embedder->Close());
}

TEST_F(EmbedderTest, SucceedsWithMobileBertAndDifferentThemes) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kMobileBert);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result0,
      text_embedder->Embed("When you go to this restaurant, they hold the "
                           "pancake upside-down before they hand it "
                           "to you. It's a great gimmick."));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result1,
      text_embedder->Embed(
          "Let's make a plan to steal the declaration of independence."));

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  // TODO: These similarity should likely be lower
#ifdef _WIN32
  EXPECT_NEAR(similarity, 0.98152, kSimilarityTolerancy);
#else
  EXPECT_NEAR(similarity, 0.95016, kSimilarityTolerancy);
#endif  // _WIN32

  MP_ASSERT_OK(text_embedder->Close());
}

TEST_F(EmbedderTest, SucceedsWithUSEAndDifferentThemes) {
  auto options = std::make_unique<TextEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kUniversalSentenceEncoderModel);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextEmbedder> text_embedder,
                          TextEmbedder::Create(std::move(options)));

  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result0,
      text_embedder->Embed("When you go to this restaurant, they hold the "
                           "pancake upside-down before they hand it "
                           "to you. It's a great gimmick."));
  MP_ASSERT_OK_AND_ASSIGN(
      TextEmbedderResult result1,
      text_embedder->Embed(
          "Let's make a plan to steal the declaration of independence."));

  // Check cosine similarity.
  MP_ASSERT_OK_AND_ASSIGN(
      double similarity, TextEmbedder::CosineSimilarity(result0.embeddings[0],
                                                        result1.embeddings[0]));
  EXPECT_NEAR(similarity, 0.780334, kSimilarityTolerancy);

  MP_ASSERT_OK(text_embedder->Close());
}

}  // namespace
}  // namespace mediapipe::tasks::text::text_embedder
