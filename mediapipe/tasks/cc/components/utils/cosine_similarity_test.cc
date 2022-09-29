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

#include "mediapipe/tasks/cc/components/utils/cosine_similarity.h"

#include <cstdint>
#include <string>
#include <vector>

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"

namespace mediapipe {
namespace tasks {
namespace components {
namespace utils {
namespace {

using ::mediapipe::tasks::components::containers::proto::EmbeddingEntry;
using ::testing::HasSubstr;

// Helper function to generate float EmbeddingEntry.
EmbeddingEntry BuildFloatEntry(std::vector<float> values) {
  EmbeddingEntry entry;
  for (const float value : values) {
    entry.mutable_float_embedding()->add_values(value);
  }
  return entry;
}

// Helper function to generate quantized EmbeddingEntry.
EmbeddingEntry BuildQuantizedEntry(std::vector<int8_t> values) {
  EmbeddingEntry entry;
  entry.mutable_quantized_embedding()->set_values(
      reinterpret_cast<uint8_t*>(values.data()), values.size());
  return entry;
}

TEST(CosineSimilarity, FailsWithQuantizedAndFloatEmbeddings) {
  auto u = BuildFloatEntry({0.1, 0.2});
  auto v = BuildQuantizedEntry({0, 1});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              HasSubstr("Cannot compute cosine similarity between quantized "
                        "and float embeddings"));
}

TEST(CosineSimilarity, FailsWithZeroNorm) {
  auto u = BuildFloatEntry({0.1, 0.2});
  auto v = BuildFloatEntry({0.0, 0.0});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      status.status().message(),
      HasSubstr("Cannot compute cosine similarity on embedding with 0 norm"));
}

TEST(CosineSimilarity, FailsWithDifferentSizes) {
  auto u = BuildFloatEntry({0.1, 0.2});
  auto v = BuildFloatEntry({0.1, 0.2, 0.3});

  auto status = CosineSimilarity(u, v);

  EXPECT_EQ(status.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.status().message(),
              HasSubstr("Cannot compute cosine similarity between embeddings "
                        "of different sizes"));
}

TEST(CosineSimilarity, SucceedsWithFloatEntries) {
  auto u = BuildFloatEntry({1.0, 0.0, 0.0, 0.0});
  auto v = BuildFloatEntry({0.5, 0.5, 0.5, 0.5});

  MP_ASSERT_OK_AND_ASSIGN(auto result, CosineSimilarity(u, v));

  EXPECT_EQ(result, 0.5);
}

TEST(CosineSimilarity, SucceedsWithQuantizedEntries) {
  auto u = BuildQuantizedEntry({127, 0, 0, 0});
  auto v = BuildQuantizedEntry({-128, 0, 0, 0});

  MP_ASSERT_OK_AND_ASSIGN(auto result, CosineSimilarity(u, v));

  EXPECT_EQ(result, -1);
}

}  // namespace
}  // namespace utils
}  // namespace components
}  // namespace tasks
}  // namespace mediapipe
