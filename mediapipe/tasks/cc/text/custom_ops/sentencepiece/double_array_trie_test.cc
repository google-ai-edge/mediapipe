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

#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/double_array_trie.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/double_array_trie_builder.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/encoder_config_generated.h"
#include "mediapipe/tasks/cc/text/custom_ops/sentencepiece/utils.h"

namespace mediapipe::tflite_operations::sentencepiece {

TEST(DoubleArrayTrieTest, Match) {
  flatbuffers::FlatBufferBuilder builder(1024);
  const std::vector<std::string> test_strings = {"A", "AAX", "AA", "B"};
  const auto trie_vector = builder.CreateVector(BuildTrie(test_strings));
  TrieBuilder trie_builder(builder);
  trie_builder.add_nodes(trie_vector);
  const auto pieces = trie_builder.Finish();
  EncoderConfigBuilder ecb(builder);
  ecb.add_pieces(pieces);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  const EncoderConfig* config = GetEncoderConfig(builder.GetBufferPointer());
  DoubleArrayTrie dat(config->pieces()->nodes());
  EXPECT_EQ(dat.LongestPrefixMatch(utils::string_view("AAL")),
            DoubleArrayTrie::Match(2, 2));

  std::vector<DoubleArrayTrie::Match> matches;
  dat.IteratePrefixMatches(
      utils::string_view("AAXL"),
      [&matches](const DoubleArrayTrie::Match& m) { matches.push_back(m); });
  EXPECT_THAT(matches, testing::ElementsAre(DoubleArrayTrie::Match(0, 1),
                                            DoubleArrayTrie::Match(2, 2),
                                            DoubleArrayTrie::Match(1, 3)));
}

TEST(DoubleArrayTrieTest, ComplexMatch) {
  flatbuffers::FlatBufferBuilder builder(1024);
  const std::vector<std::string> test_strings = {"\xe2\x96\x81the", ",", "s",
                                                 "\xe2\x96\x81Hello"};
  const std::vector<int> test_ids = {0, 5, 10, 15};
  const auto trie_vector =
      builder.CreateVector(BuildTrie(test_strings, test_ids));
  TrieBuilder trie_builder(builder);
  trie_builder.add_nodes(trie_vector);
  const auto pieces = trie_builder.Finish();
  EncoderConfigBuilder ecb(builder);
  ecb.add_pieces(pieces);
  FinishEncoderConfigBuffer(builder, ecb.Finish());
  const EncoderConfig* config = GetEncoderConfig(builder.GetBufferPointer());
  DoubleArrayTrie dat(config->pieces()->nodes());

  std::vector<DoubleArrayTrie::Match> matches;
  dat.IteratePrefixMatches(
      utils::string_view("\xe2\x96\x81Hello"),
      [&matches](const DoubleArrayTrie::Match& m) { matches.push_back(m); });
  EXPECT_THAT(matches, testing::ElementsAre(DoubleArrayTrie::Match(15, 8)));
}

}  // namespace mediapipe::tflite_operations::sentencepiece
