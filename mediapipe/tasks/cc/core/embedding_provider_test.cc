/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/core/embedding_provider.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::core {
namespace {

class FakeEmbeddingProvider : public EmbeddingProvider {
 public:
  absl::StatusOr<std::optional<std::vector<float>>> EmbedContent(
      const std::vector<TaskPart>& content) override {
    if (content.empty()) {
      return std::nullopt;
    }
    std::vector<float> mock_embedding;
    for (const auto& part : content) {
      if (std::holds_alternative<TextPart>(part)) {
        mock_embedding.push_back(
            static_cast<float>(std::get<TextPart>(part).text.length()));
      } else if (std::holds_alternative<ImagePart>(part)) {
        mock_embedding.push_back(
            static_cast<float>(std::get<ImagePart>(part).image_bytes.length()));
      } else if (std::holds_alternative<AudioPart>(part)) {
        mock_embedding.push_back(
            static_cast<float>(std::get<AudioPart>(part).audio_data.size()));
      }
    }
    return mock_embedding;
  }
};

TEST(EmbeddingProviderTest, TestFakeProviderAndParts) {
  FakeEmbeddingProvider provider;

  std::vector<TaskPart> content;
  content.push_back(TextPart{"hello"});
  content.push_back(ImagePart{"bytes"});
  content.push_back(AudioPart{std::vector<float>{1.0f, 2.0f}});

  MP_ASSERT_OK_AND_ASSIGN(auto embedding_opt, provider.EmbedContent(content));
  ASSERT_TRUE(embedding_opt.has_value());
  auto embedding = *embedding_opt;
  ASSERT_EQ(embedding.size(), 3);
  EXPECT_EQ(embedding[0], 5.0f);
  EXPECT_EQ(embedding[1], 5.0f);
  EXPECT_EQ(embedding[2], 2.0f);
}

TEST(EmbeddingProviderTest, TestEmptyContentReturnsNullopt) {
  FakeEmbeddingProvider provider;
  std::vector<TaskPart> content;
  MP_ASSERT_OK_AND_ASSIGN(auto embedding_opt, provider.EmbedContent(content));
  EXPECT_FALSE(embedding_opt.has_value());
}

}  // namespace
}  // namespace mediapipe::tasks::core
