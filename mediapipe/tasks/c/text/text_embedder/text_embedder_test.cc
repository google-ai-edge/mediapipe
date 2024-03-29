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

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestBertModelPath[] =
    "mobilebert_embedding_with_metadata.tflite";
constexpr char kTestString0[] =
    "When you go to this restaurant, they hold the pancake upside-down "
    "before they hand it to you. It's a great gimmick.";
constexpr char kTestString1[] =
    "Let's make a plan to steal the declaration of independence.";
constexpr float kPrecision = 1e-3;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(TextEmbedderTest, SmokeTest) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* embedder_options= */
      {/* l2_normalize= */ false, /* quantize= */ true},
  };

  void* embedder = text_embedder_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  TextEmbedderResult result;
  text_embedder_embed(embedder, kTestString0, &result, /* error_msg */ nullptr);
  EXPECT_EQ(result.embeddings_count, 1);
  EXPECT_EQ(result.embeddings[0].values_count, 512);

  text_embedder_close_result(&result);
  text_embedder_close(embedder, /* error_msg */ nullptr);
}

TEST(TextEmbedderTest, SucceedsWithCosineSimilarity) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* embedder_options= */
      {/* l2_normalize= */ false,
       /* quantize= */ false}};

  void* embedder = text_embedder_create(&options,
                                        /* error_msg */ nullptr);
  EXPECT_NE(embedder, nullptr);

  // Extract both embeddings.
  TextEmbedderResult result0;
  text_embedder_embed(embedder, kTestString0, &result0,
                      /* error_msg */ nullptr);
  TextEmbedderResult result1;
  text_embedder_embed(embedder, kTestString1, &result1,
                      /* error_msg */ nullptr);

  // Check cosine similarity.
  double similarity;
  text_embedder_cosine_similarity(&result0.embeddings[0],
                                  &result1.embeddings[0], &similarity, nullptr);
  double expected_similarity = 0.98077;
  EXPECT_LE(abs(similarity - expected_similarity), kPrecision);

  text_embedder_close_result(&result0);
  text_embedder_close_result(&result1);
  text_embedder_close(embedder, /* error_msg */ nullptr);
}

TEST(TextEmbedderTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  TextEmbedderOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* embedder_options= */ {},
  };

  char* error_msg;
  void* embedder = text_embedder_create(&options, &error_msg);
  EXPECT_EQ(embedder, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("INVALID_ARGUMENT"));

  free(error_msg);
}

}  // namespace
