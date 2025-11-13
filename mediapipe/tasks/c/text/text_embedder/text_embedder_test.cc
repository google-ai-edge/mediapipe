/* Copyright 2023 The MediaPipe Authors.

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

#include "mediapipe/tasks/c/text/text_embedder/text_embedder.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestBertModelPath[] =
    "mobilebert_embedding_with_metadata.tflite";
constexpr char kTestString0[] =
    "When you go to this restaurant, they hold the pancake upside-down "
    "before they hand it to you. It's a great gimmick.";
constexpr char kTestString1[] =
    "Let's make a plan to steal the declaration of independence.";
constexpr float kCosineSimilarityThreshold = 0.95;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(TextEmbedderTest, SmokeTest) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextEmbedderOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .embedder_options = {.l2_normalize = false, .quantize = true},
  };

  MpTextEmbedderPtr embedder;
  EXPECT_EQ(MpTextEmbedderCreate(&options, &embedder), kMpOk);
  EXPECT_NE(embedder, nullptr);

  TextEmbedderResult result;
  EXPECT_EQ(MpTextEmbedderEmbed(embedder, kTestString0, &result), kMpOk);
  EXPECT_EQ(result.embeddings_count, 1);
  EXPECT_EQ(result.embeddings[0].values_count, 512);

  MpTextEmbedderCloseResult(&result);
  EXPECT_EQ(result.embeddings, nullptr);
  EXPECT_EQ(MpTextEmbedderClose(embedder), kMpOk);
}

TEST(TextEmbedderTest, SucceedsWithCosineSimilarity) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextEmbedderOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .embedder_options = {.l2_normalize = false, .quantize = false}};

  MpTextEmbedderPtr embedder;
  EXPECT_EQ(MpTextEmbedderCreate(&options, &embedder), kMpOk);
  EXPECT_NE(embedder, nullptr);

  // Extract both embeddings.
  TextEmbedderResult result0;
  EXPECT_EQ(MpTextEmbedderEmbed(embedder, kTestString0, &result0), kMpOk);
  TextEmbedderResult result1;
  EXPECT_EQ(MpTextEmbedderEmbed(embedder, kTestString1, &result1), kMpOk);

  ASSERT_EQ(result0.embeddings_count, 1);
  ASSERT_EQ(result1.embeddings_count, 1);

  // Check cosine similarity.
  double similarity;
  EXPECT_EQ(MpTextEmbedderCosSimilarity(&result0.embeddings[0],
                                        &result1.embeddings[0], &similarity),
            kMpOk);
  EXPECT_GE(similarity, kCosineSimilarityThreshold);

  MpTextEmbedderCloseResult(&result0);
  EXPECT_EQ(result0.embeddings, nullptr);
  MpTextEmbedderCloseResult(&result1);
  EXPECT_EQ(result1.embeddings, nullptr);
  EXPECT_EQ(MpTextEmbedderClose(embedder), kMpOk);
}

TEST(TextEmbedderTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  TextEmbedderOptions options = {
      .base_options = {.model_asset_path = nullptr},
      .embedder_options = {},
  };

  MpTextEmbedderPtr embedder;
  EXPECT_EQ(MpTextEmbedderCreate(&options, &embedder), kMpInvalidArgument);
}

}  // namespace
