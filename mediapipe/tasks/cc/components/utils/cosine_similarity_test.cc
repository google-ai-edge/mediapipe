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

#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"

#include <cstdint>
#include <string>
#include <vector>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace utils {
namespace {

using ::mediapipe::tasks::components::containers::Embedding;
using ::testing::HasSubstr;

// Helper function to generate float Embedding.
Embedding BuildFloatEmbedding(std::vector<float> values) {
  Embedding embedding;
  embedding.float_embedding = values;
  return embedding;
}

// Helper function to generate quantized Embedding.
Embedding BuildQuantizedEmbedding(std::vector<int8_t> values) {
  Embedding embedding;
  uint8_t* data = reinterpret_cast<uint8_t*>(values.data());
  embedding.quantized_embedding = {data, data + values.size()};
  return embedding;
}

TEST(CosineSimilarity, FailsWithQuantizedAndFloatEmbeddings) {
  auto u = BuildFloatEmbedding({0.1, 0.2});
  auto v = BuildQuantizedEmbedding({0, 1});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              HasSubstr("Cannot compute cosine similarity between quantized "
                        "and float embeddings"));
}

TEST(CosineSimilarity, FailsWithZeroNorm) {
  auto u = BuildFloatEmbedding({0.1, 0.2});
  auto v = BuildFloatEmbedding({0.0, 0.0});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.status().message(),
      HasSubstr("Cannot compute cosine similarity on embedding with 0 norm"));
}

TEST(CosineSimilarity, FailsWithDifferentSizes) {
  auto u = BuildFloatEmbedding({0.1, 0.2});
  auto v = BuildFloatEmbedding({0.1, 0.2, 0.3});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              HasSubstr("Cannot compute cosine similarity between embeddings "
                        "of different sizes"));
}

TEST(CosineSimilarity, SucceedsWithFloatEntries) {
  auto u = BuildFloatEmbedding({1.0, 0.0, 0.0, 0.0});
  auto v = BuildFloatEmbedding({0.5, 0.5, 0.5, 0.5});

  MP_ASSERT_OK_AND_ASSIGN(auto result, CosineSimilarity(u, v));

  EXPECT_EQ(result, 0.5);
}

TEST(CosineSimilarity, SucceedsWithQuantizedEntries) {
  auto u = BuildQuantizedEmbedding({127, 0, 0, 0});
  auto v = BuildQuantizedEmbedding({-128, 0, 0, 0});

  MP_ASSERT_OK_AND_ASSIGN(auto result, CosineSimilarity(u, v));

  EXPECT_EQ(result, -1);
}

}  // namespace
}  // namespace utils
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
