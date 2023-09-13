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

#include "mediapipe/tasks/cc/text/utils/vocab_utils.h"

#include "absl/container/node_hash_map.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/core/utils.h"

namespace mediapipe {
namespace tasks {
namespace text {

using ::mediapipe::tasks::core::LoadBinaryContent;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

namespace {
constexpr char kVocabPath[] = "mediapipe/tasks/testdata/text/vocab.txt";
constexpr char kVocabAndIndexPath[] =
    "mediapipe/tasks/testdata/text/vocab_with_index.txt";

}  // namespace

TEST(CommonUtilsTest, TestLoadVocabFromFile) {
  std::vector<std::string> vocab = LoadVocabFromFile(kVocabPath);

  EXPECT_THAT(vocab, UnorderedElementsAre("token1", "token2", "token3"));
}

TEST(CommonUtilsTest, TestLoadVocabFromBuffer) {
  std::string buffer = LoadBinaryContent(kVocabPath);
  std::vector<std::string> vocab =
      LoadVocabFromBuffer(buffer.data(), buffer.size());

  EXPECT_THAT(vocab, UnorderedElementsAre("token1", "token2", "token3"));
}

TEST(CommonUtilsTest, TestLoadVocabAndIndexFromFile) {
  absl::node_hash_map<std::string, int> vocab =
      LoadVocabAndIndexFromFile(kVocabAndIndexPath);

  EXPECT_THAT(vocab, UnorderedElementsAre(Pair("token1", 0), Pair("token2", 1),
                                          Pair("token3", 2)));
}

TEST(CommonUtilsTest, TestLoadVocabAndIndexFromBuffer) {
  std::string buffer = LoadBinaryContent(kVocabAndIndexPath);
  absl::node_hash_map<std::string, int> vocab =
      LoadVocabAndIndexFromBuffer(buffer.data(), buffer.size());

  EXPECT_THAT(vocab, UnorderedElementsAre(Pair("token1", 0), Pair("token2", 1),
                                          Pair("token3", 2)));
}

}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
