// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/util/sequence/media_sequence_util.h"

#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace mediapipe {
namespace mediasequence {
namespace {

BYTES_CONTEXT_FEATURE(StringFeature, "string_feature");
INT64_CONTEXT_FEATURE(Int64Feature, "int64_feature");
FLOAT_CONTEXT_FEATURE(FloatFeature, "float_feature");
VECTOR_BYTES_CONTEXT_FEATURE(StringVectorFeature, "string_vector_feature");
VECTOR_INT64_CONTEXT_FEATURE(Int64VectorFeature, "int64_vector_feature");
VECTOR_FLOAT_CONTEXT_FEATURE(FloatVectorFeature, "float_vector_feature");
BYTES_FEATURE_LIST(StringFeatureList, "string_feature_list");
INT64_FEATURE_LIST(Int64FeatureList, "int64_feature_list");
FLOAT_FEATURE_LIST(FloatFeatureList, "float_feature_list");
VECTOR_BYTES_FEATURE_LIST(VectorStringFeatureList,
                          "vector_string_feature_list");
VECTOR_INT64_FEATURE_LIST(VectorInt64FeatureList, "vector_int64_feature_list");
VECTOR_FLOAT_FEATURE_LIST(VectorFloatFeatureList, "vector_float_feature_list");

// Testing this checks for name collisions and compiler errors.
FIXED_PREFIX_BYTES_CONTEXT_FEATURE(OneStringFeature, "string_feature", "ONE");
FIXED_PREFIX_BYTES_CONTEXT_FEATURE(TwoStringFeature, "string_feature", "TWO");
FIXED_PREFIX_INT64_CONTEXT_FEATURE(OneInt64Feature, "int64_feature", "ONE");
FIXED_PREFIX_INT64_CONTEXT_FEATURE(TwoInt64Feature, "int64_feature", "TWO");
FIXED_PREFIX_FLOAT_CONTEXT_FEATURE(OneFloatFeature, "float_feature", "ONE");
FIXED_PREFIX_FLOAT_CONTEXT_FEATURE(TwoFloatFeature, "float_feature", "TWO");
FIXED_PREFIX_VECTOR_BYTES_CONTEXT_FEATURE(OneStringVectorFeature,
                                          "string_vector_feature", "ONE");
FIXED_PREFIX_VECTOR_BYTES_CONTEXT_FEATURE(TwoStringVectorFeature,
                                          "string_vector_feature", "TWO");
FIXED_PREFIX_VECTOR_INT64_CONTEXT_FEATURE(OneInt64VectorFeature,
                                          "int64_vector_feature", "ONE");
FIXED_PREFIX_VECTOR_INT64_CONTEXT_FEATURE(TwoInt64VectorFeature,
                                          "int64_vector_feature", "TWO");
FIXED_PREFIX_VECTOR_FLOAT_CONTEXT_FEATURE(OneFloatVectorFeature,
                                          "float_vector_feature", "ONE");
FIXED_PREFIX_VECTOR_FLOAT_CONTEXT_FEATURE(TwoFloatVectorFeature,
                                          "float_vector_feature", "TWO");
FIXED_PREFIX_BYTES_FEATURE_LIST(OneStringFeatureList, "string_feature_list",
                                "ONE");
FIXED_PREFIX_BYTES_FEATURE_LIST(TwoStringFeatureList, "string_feature_list",
                                "TWO");
FIXED_PREFIX_INT64_FEATURE_LIST(OneInt64FeatureList, "int64_feature_list",
                                "ONE");
FIXED_PREFIX_INT64_FEATURE_LIST(TwoInt64FeatureList, "int64_feature_list",
                                "TWO");
FIXED_PREFIX_FLOAT_FEATURE_LIST(OneFloatFeatureList, "float_feature_list",
                                "ONE");
FIXED_PREFIX_FLOAT_FEATURE_LIST(TwoFloatFeatureList, "float_feature_list",
                                "TWO");
FIXED_PREFIX_VECTOR_BYTES_FEATURE_LIST(OneVectorStringFeatureList,
                                       "vector_string_feature_list", "ONE");
FIXED_PREFIX_VECTOR_BYTES_FEATURE_LIST(TwoVectorStringFeatureList,
                                       "vector_string_feature_list", "TWO");
FIXED_PREFIX_VECTOR_INT64_FEATURE_LIST(OneVectorInt64FeatureList,
                                       "vector_int64_feature_list", "ONE");
FIXED_PREFIX_VECTOR_INT64_FEATURE_LIST(TwoVectorInt64FeatureList,
                                       "vector_int64_feature_list", "TWO");
FIXED_PREFIX_VECTOR_FLOAT_FEATURE_LIST(OneVectorFloatFeatureList,
                                       "vector_float_feature_list", "ONE");
FIXED_PREFIX_VECTOR_FLOAT_FEATURE_LIST(TwoVectorFloatFeatureList,
                                       "vector_float_feature_list", "TWO");

// This checks for compiler errors.
PREFIXED_VECTOR_BYTES_CONTEXT_FEATURE(AnyStringFeature,
                                      "string_vector_feature");

// Adapted from third_party/tensorflow/core/example/example.proto:
//
// Below is a SequenceExample for a movie recommendation application recording a
// sequence of ratings by a user. The time-independent features ("locale",
// "age", "favorites") describing the user are part of the context. The sequence
// of movies the user rated are part of the feature_lists. For each movie in the
// sequence we have information on its name and actors and the user's rating.
// This information is recorded in three separate feature_list(s).
// In the example below there are only two movies. All three feature_list(s),
// namely "movie_ratings", "movie_names", and "actors" have a feature value for
// both movies. Note, that "actors" is itself a bytes_list with multiple
// strings per movie.
const char kAsciiSequenceExample[] =
    "context: {"
    "  feature: {"
    "    key  : 'locale'"
    "    value: {"
    "      bytes_list: {"
    "        value: [ 'pt_BR' ]"
    "      }"
    "    }"
    "  }"
    "  feature: {"
    "    key  : 'age'"
    "    value: {"
    "      float_list: {"
    "        value: [ 19.0 ]"
    "      }"
    "    }"
    "  }"
    "  feature: {"
    "    key  : 'favorites'"
    "    value: {"
    "      bytes_list: {"
    "        value: [ 'Majesty Rose', 'Savannah Outen', 'One Direction' ]"
    "      }"
    "    }"
    "  }"
    "}"
    "feature_lists: {"
    "  feature_list: {"
    "    key  : 'movie_ratings'"
    "    value: {"
    "      feature: {"
    "        float_list: {"
    "          value: [ 4.5 ]"
    "        }"
    "      }"
    "      feature: {"
    "        float_list: {"
    "          value: [ 5.0, 2.3 ]"
    "        }"
    "      }"
    "    }"
    "  }"
    "  feature_list: {"
    "    key  : 'runtimes'"
    "    value: {"
    "      feature: {"
    "        int64_list: {"
    "          value: [ 123, 84 ]"
    "        }"
    "      }"
    "      feature: {"
    "        int64_list: {"
    "          value: [ 97 ]"
    "        }"
    "      }"
    "    }"
    "  }"
    "  feature_list: {"
    "    key  : 'movie_names'"
    "    value: {"
    "      feature: {"
    "        bytes_list: {"
    "          value: [ 'The Shawshank Redemption' ]"
    "        }"
    "      }"
    "      feature: {"
    "        bytes_list: {"
    "          value: [ 'Fight Club']"
    "        }"
    "      }"
    "    }"
    "  }"
    "  feature_list: {"
    "    key  : 'actors'"
    "    value: {"
    "      feature: {"
    "        bytes_list: {"
    "          value: [ 'Tim Robbins', 'Morgan Freeman' ]"
    "        }"
    "      }"
    "      feature: {"
    "        bytes_list: {"
    "          value: [ 'Brad Pitt', 'Edward Norton', 'Helena Bonham Carter' ]"
    "        }"
    "      }"
    "    }"
    "  }"
    "}";

class MediaSequenceUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    sequence_example_ =
        mediapipe::ParseTextProtoOrDie<tensorflow::SequenceExample>(
            kAsciiSequenceExample);
  }

  tensorflow::SequenceExample sequence_example_;
};

TEST_F(MediaSequenceUtilTest, GetFeatureList) {
  const tensorflow::FeatureList& fl =
      GetFeatureList(sequence_example_, "movie_names");
  ASSERT_EQ(2, fl.feature_size());
  EXPECT_EQ("Fight Club", fl.feature().Get(1).bytes_list().value(0));
}

TEST_F(MediaSequenceUtilTest, GetFloatsAt) {
  const auto& ratings0 = GetFloatsAt(sequence_example_, "movie_ratings", 0);
  ASSERT_EQ(1, ratings0.size());
  EXPECT_FLOAT_EQ(4.5, ratings0[0]);
  const auto& ratings1 = GetFloatsAt(sequence_example_, "movie_ratings", 1);
  ASSERT_EQ(2, ratings1.size());
  EXPECT_FLOAT_EQ(5.0, ratings1[0]);
  EXPECT_FLOAT_EQ(2.3, ratings1[1]);
}

TEST_F(MediaSequenceUtilTest, GetInt64sAt) {
  const auto& runtimes0 = GetInt64sAt(sequence_example_, "runtimes", 0);
  ASSERT_EQ(2, runtimes0.size());
  EXPECT_EQ(123, runtimes0[0]);
  EXPECT_EQ(84, runtimes0[1]);
  const auto& runtimes1 = GetInt64sAt(sequence_example_, "runtimes", 1);
  ASSERT_EQ(1, runtimes1.size());
  EXPECT_EQ(97, runtimes1[0]);
}

TEST_F(MediaSequenceUtilTest, GetBytesAt) {
  const auto& actors0 = GetBytesAt(sequence_example_, "actors", 0);
  ASSERT_EQ(2, actors0.size());
  EXPECT_EQ("Tim Robbins", actors0[0]);
  EXPECT_EQ("Morgan Freeman", actors0[1]);
  const auto& actors1 = GetBytesAt(sequence_example_, "actors", 1);
  ASSERT_EQ(3, actors1.size());
  EXPECT_EQ("Brad Pitt", actors1[0]);
  EXPECT_EQ("Edward Norton", actors1[1]);
  EXPECT_EQ("Helena Bonham Carter", actors1[2]);
}

TEST_F(MediaSequenceUtilTest, RoundTripFloatList) {
  tensorflow::SequenceExample sequence_example;
  std::string key = "key";
  std::vector<float> expected_values{1.0f, 3.0f};
  AddFloatContainer(key, expected_values, &sequence_example);
  auto values = GetFloatsAt(sequence_example, key, 0);
  ASSERT_EQ(expected_values.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(expected_values[i], values[i]);
  }
}

TEST_F(MediaSequenceUtilTest, RoundTripInt64List) {
  tensorflow::SequenceExample sequence_example;
  std::string key = "key";
  std::vector<int64> expected_values{1, 3};
  AddInt64Container(key, expected_values, &sequence_example);
  auto values = GetInt64sAt(sequence_example, key, 0);
  ASSERT_EQ(expected_values.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(expected_values[i], values[i]);
  }
}

TEST_F(MediaSequenceUtilTest, RoundTripBytesList) {
  tensorflow::SequenceExample sequence_example;
  std::string key = "key";
  std::vector<std::string> expected_values{"1", "3"};
  AddBytesContainer(key, expected_values, &sequence_example);
  auto values = GetBytesAt(sequence_example, key, 0);
  ASSERT_EQ(expected_values.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_EQ(expected_values[i], values[i]);
  }
}

TEST_F(MediaSequenceUtilTest, RoundTripContextFeature) {
  tensorflow::SequenceExample sequence_example;
  std::string video_id_key = "video_id";
  std::string video_id = "test";
  MutableContext(video_id_key, &sequence_example)
      ->mutable_bytes_list()
      ->add_value(video_id);

  std::string result =
      GetContext(sequence_example, video_id_key).bytes_list().value(0);
  ASSERT_EQ(result, video_id);
}

TEST_F(MediaSequenceUtilTest, RoundTripContextFeatureList) {
  tensorflow::SequenceExample sequence_example;
  // Test context float list.
  std::string clip_label_score_key = "clip_label_score";
  std::vector<float> clip_label_scores{0.5, 0.8};
  SetContextFloatList(clip_label_score_key, clip_label_scores,
                      &sequence_example);
  for (int i = 0; i < clip_label_scores.size(); ++i) {
    ASSERT_FLOAT_EQ(clip_label_scores[i],
                    GetContext(sequence_example, clip_label_score_key)
                        .float_list()
                        .value(i));
  }
  // Test context in64 list.
  std::string clip_label_index_key = "clip_label_index";
  std::vector<int64> clip_label_indices{2, 0};
  SetContextInt64List(clip_label_index_key, clip_label_indices,
                      &sequence_example);
  for (int i = 0; i < clip_label_indices.size(); ++i) {
    ASSERT_EQ(clip_label_indices[i],
              GetContext(sequence_example, clip_label_index_key)
                  .int64_list()
                  .value(i));
  }
  // Test context bytes list.
  std::string clip_label_string_key = "clip_label_string";
  std::vector<std::string> clip_label_strings{"run", "sit"};
  SetContextBytesList(clip_label_string_key, clip_label_strings,
                      &sequence_example);
  for (int i = 0; i < clip_label_strings.size(); ++i) {
    ASSERT_EQ(clip_label_strings[i],
              GetContext(sequence_example, clip_label_string_key)
                  .bytes_list()
                  .value(i));
  }
}

TEST_F(MediaSequenceUtilTest, ContextKeyMissing) {
  tensorflow::SequenceExample sequence_example;
  ASSERT_DEATH({ GetContext(sequence_example, "key/is/unavailable"); },
               "Could not find context key key/is/unavailable");
}

TEST_F(MediaSequenceUtilTest, RoundTripFeatureListsFeature) {
  tensorflow::SequenceExample sequence_example;
  std::string timestamp_key = "timestamp";
  int64 timestamp = 1000;
  MutableFeatureList(timestamp_key, &sequence_example)
      ->add_feature()
      ->mutable_int64_list()
      ->add_value(timestamp);
  MutableFeatureList(timestamp_key, &sequence_example)
      ->add_feature()
      ->mutable_int64_list()
      ->add_value(timestamp * 2);

  const auto& result_1 = GetInt64sAt(sequence_example, timestamp_key, 0);
  const auto& result_2 = GetInt64sAt(sequence_example, timestamp_key, 1);
  ASSERT_EQ(result_1[0], timestamp);
  ASSERT_EQ(result_2[0], timestamp * 2);
}

TEST_F(MediaSequenceUtilTest, HasContext) {
  ASSERT_TRUE(HasContext(sequence_example_, "locale"));
  ASSERT_FALSE(HasContext(sequence_example_, "garbage_key"));
}

TEST_F(MediaSequenceUtilTest, HasFeatureList) {
  ASSERT_TRUE(HasFeatureList(sequence_example_, "movie_names"));
  ASSERT_FALSE(HasFeatureList(sequence_example_, "garbage_key"));
}

TEST_F(MediaSequenceUtilTest, SetContextFloat) {
  tensorflow::SequenceExample example;
  std::string key = "test";
  ASSERT_DEATH({ GetContext(example, key); },
               "Could not find context key " + key);
  SetContextFloat(key, 1.0, &example);
  ASSERT_EQ(GetContext(example, key).float_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).float_list().value(0), 1.0);
  SetContextFloat(key, 2.0, &example);
  ASSERT_EQ(GetContext(example, key).float_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).float_list().value(0), 2.0);
}

TEST_F(MediaSequenceUtilTest, SetContextInt64) {
  tensorflow::SequenceExample example;
  std::string key = "test";
  ASSERT_DEATH({ GetContext(example, key); },
               "Could not find context key " + key);
  SetContextInt64(key, 1, &example);
  ASSERT_EQ(GetContext(example, key).int64_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).int64_list().value(0), 1);
  SetContextInt64(key, 2, &example);
  ASSERT_EQ(GetContext(example, key).int64_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).int64_list().value(0), 2);
}

TEST_F(MediaSequenceUtilTest, SetContextBytes) {
  tensorflow::SequenceExample example;
  std::string key = "test";
  ASSERT_DEATH({ GetContext(example, key); },
               "Could not find context key " + key);
  SetContextBytes(key, "one", &example);
  ASSERT_EQ(GetContext(example, key).bytes_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).bytes_list().value(0), "one");
  SetContextBytes(key, "two", &example);
  ASSERT_EQ(GetContext(example, key).bytes_list().value_size(), 1);
  ASSERT_EQ(GetContext(example, key).bytes_list().value(0), "two");
}

TEST_F(MediaSequenceUtilTest, StringFeature) {
  tensorflow::SequenceExample example;
  std::string test_value = "string";

  ASSERT_FALSE(HasStringFeature(example));
  SetStringFeature(test_value, &example);
  ASSERT_TRUE(HasStringFeature(example));
  ASSERT_EQ(test_value, GetStringFeature(example));
  ClearStringFeature(&example);
  ASSERT_FALSE(HasStringFeature(example));
  ASSERT_EQ(GetStringFeatureKey(), "string_feature");
}

TEST_F(MediaSequenceUtilTest, Int64Feature) {
  tensorflow::SequenceExample example;
  int64 test_value = 47;

  ASSERT_FALSE(HasInt64Feature(example));
  SetInt64Feature(test_value, &example);
  ASSERT_TRUE(HasInt64Feature(example));
  ASSERT_EQ(test_value, GetInt64Feature(example));
  ClearInt64Feature(&example);
  ASSERT_FALSE(HasInt64Feature(example));
  ASSERT_EQ(GetInt64FeatureKey(), "int64_feature");
}

TEST_F(MediaSequenceUtilTest, FloatFeature) {
  tensorflow::SequenceExample example;
  int64 test_value = 47.0f;

  ASSERT_FALSE(HasFloatFeature(example));
  SetFloatFeature(test_value, &example);
  ASSERT_TRUE(HasFloatFeature(example));
  ASSERT_EQ(test_value, GetFloatFeature(example));
  ClearFloatFeature(&example);
  ASSERT_FALSE(HasFloatFeature(example));
  ASSERT_EQ(GetFloatFeatureKey(), "float_feature");
}

TEST_F(MediaSequenceUtilTest, StringVectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<std::string> test_value = {"string1", "string2"};

  ASSERT_FALSE(HasStringVectorFeature(example));
  ASSERT_EQ(0, GetStringVectorFeatureSize(example));
  SetStringVectorFeature(test_value, &example);
  ASSERT_EQ(test_value.size(), GetStringVectorFeatureSize(example));
  ASSERT_TRUE(HasStringVectorFeature(example));
  ASSERT_THAT(GetStringVectorFeature(example),
              testing::ElementsAreArray(test_value));
  AddStringVectorFeature(test_value[0], &example);
  AddStringVectorFeature(test_value[1], &example);
  ASSERT_EQ(test_value.size() * 2, GetStringVectorFeatureSize(example));
  ClearStringVectorFeature(&example);
  ASSERT_FALSE(HasStringVectorFeature(example));
  AddStringVectorFeature(test_value[0], &example);
  AddStringVectorFeature(test_value[1], &example);
  ASSERT_THAT(GetStringVectorFeature(example),
              testing::ElementsAreArray(test_value));
  ASSERT_EQ(test_value[1], GetStringVectorFeatureAt(example, 1));
  SetStringVectorFeature({"compile", "test"}, &example);
  ASSERT_EQ(GetStringVectorFeatureKey(), "string_vector_feature");
}

TEST_F(MediaSequenceUtilTest, Int64VectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<int64> test_value = {47, 42};

  ASSERT_FALSE(HasInt64VectorFeature(example));
  ASSERT_EQ(0, GetInt64VectorFeatureSize(example));
  SetInt64VectorFeature(test_value, &example);
  ASSERT_EQ(test_value.size(), GetInt64VectorFeatureSize(example));
  ASSERT_TRUE(HasInt64VectorFeature(example));
  ASSERT_THAT(GetInt64VectorFeature(example),
              testing::ElementsAreArray(test_value));
  AddInt64VectorFeature(test_value[0], &example);
  AddInt64VectorFeature(test_value[1], &example);
  ASSERT_EQ(test_value.size() * 2, GetInt64VectorFeatureSize(example));
  ClearInt64VectorFeature(&example);
  ASSERT_FALSE(HasInt64VectorFeature(example));
  AddInt64VectorFeature(test_value[0], &example);
  AddInt64VectorFeature(test_value[1], &example);
  ASSERT_THAT(GetInt64VectorFeature(example),
              testing::ElementsAreArray(test_value));
  ASSERT_EQ(test_value[1], GetInt64VectorFeatureAt(example, 1));
  SetInt64VectorFeature({3, 5}, &example);
  ASSERT_EQ(GetInt64VectorFeatureKey(), "int64_vector_feature");
}

TEST_F(MediaSequenceUtilTest, FloatVectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<float> test_value = {47.0f, 42.0f};

  ASSERT_FALSE(HasFloatVectorFeature(example));
  ASSERT_EQ(0, GetFloatVectorFeatureSize(example));
  SetFloatVectorFeature(test_value, &example);
  ASSERT_EQ(test_value.size(), GetFloatVectorFeatureSize(example));
  ASSERT_TRUE(HasFloatVectorFeature(example));
  ASSERT_THAT(GetFloatVectorFeature(example),
              testing::ElementsAreArray(test_value));
  AddFloatVectorFeature(test_value[0], &example);
  AddFloatVectorFeature(test_value[1], &example);
  ASSERT_EQ(test_value.size() * 2, GetFloatVectorFeatureSize(example));
  ClearFloatVectorFeature(&example);
  ASSERT_FALSE(HasFloatVectorFeature(example));
  AddFloatVectorFeature(test_value[0], &example);
  AddFloatVectorFeature(test_value[1], &example);
  ASSERT_THAT(GetFloatVectorFeature(example),
              testing::ElementsAreArray(test_value));
  ASSERT_EQ(test_value[1], GetFloatVectorFeatureAt(example, 1));
  SetFloatVectorFeature({3.0f, 5.0f}, &example);
  ASSERT_EQ(GetFloatVectorFeatureKey(), "float_vector_feature");
}

TEST_F(MediaSequenceUtilTest, StringFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<std::string> test_value = {"string1", "string2"};

  ASSERT_FALSE(HasStringFeatureList(example));
  ASSERT_EQ(0, GetStringFeatureListSize(example));
  AddStringFeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetStringFeatureListAt(example, 0));
  ASSERT_EQ(1, GetStringFeatureListSize(example));
  ASSERT_TRUE(HasStringFeatureList(example));
  AddStringFeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetStringFeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetStringFeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetStringFeatureListSize(example));
  ASSERT_TRUE(HasStringFeatureList(example));
  ClearStringFeatureList(&example);
  ASSERT_FALSE(HasStringFeatureList(example));
  ASSERT_EQ(0, GetStringFeatureListSize(example));
  ASSERT_EQ(GetStringFeatureListKey(), "string_feature_list");
}

TEST_F(MediaSequenceUtilTest, Int64FeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<int64> test_value = {47, 42};

  ASSERT_FALSE(HasInt64FeatureList(example));
  ASSERT_EQ(0, GetInt64FeatureListSize(example));
  AddInt64FeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetInt64FeatureListAt(example, 0));
  ASSERT_EQ(1, GetInt64FeatureListSize(example));
  ASSERT_TRUE(HasInt64FeatureList(example));
  AddInt64FeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetInt64FeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetInt64FeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetInt64FeatureListSize(example));
  ASSERT_TRUE(HasInt64FeatureList(example));
  ClearInt64FeatureList(&example);
  ASSERT_FALSE(HasInt64FeatureList(example));
  ASSERT_EQ(0, GetInt64FeatureListSize(example));
  ASSERT_EQ(GetInt64FeatureListKey(), "int64_feature_list");
}

TEST_F(MediaSequenceUtilTest, FloatFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<float> test_value = {47.f, 42.f};

  ASSERT_FALSE(HasFloatFeatureList(example));
  ASSERT_EQ(0, GetFloatFeatureListSize(example));
  AddFloatFeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetFloatFeatureListAt(example, 0));
  ASSERT_EQ(1, GetFloatFeatureListSize(example));
  ASSERT_TRUE(HasFloatFeatureList(example));
  AddFloatFeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetFloatFeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetFloatFeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetFloatFeatureListSize(example));
  ASSERT_TRUE(HasFloatFeatureList(example));
  ClearFloatFeatureList(&example);
  ASSERT_FALSE(HasFloatFeatureList(example));
  ASSERT_EQ(0, GetFloatFeatureListSize(example));
  ASSERT_EQ(GetFloatFeatureListKey(), "float_feature_list");
}

TEST_F(MediaSequenceUtilTest, VectorStringFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<std::string>> test_value = {
      {"string1", "string2"}, {"string3", "string4"}};

  ASSERT_FALSE(HasVectorStringFeatureList(example));
  ASSERT_EQ(0, GetVectorStringFeatureListSize(example));
  AddVectorStringFeatureList(test_value[0], &example);
  ASSERT_THAT(GetVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_EQ(1, GetVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasVectorStringFeatureList(example));
  AddVectorStringFeatureList(test_value[1], &example);
  ASSERT_THAT(GetVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_THAT(GetVectorStringFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value[1]));
  ASSERT_EQ(test_value.size(), GetVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasVectorStringFeatureList(example));
  ClearVectorStringFeatureList(&example);
  ASSERT_FALSE(HasVectorStringFeatureList(example));
  ASSERT_EQ(0, GetVectorStringFeatureListSize(example));
  ASSERT_EQ(GetVectorStringFeatureListKey(), "vector_string_feature_list");
}

TEST_F(MediaSequenceUtilTest, VectorInt64FeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<int64>> test_value = {{47, 42}, {3, 5}};

  ASSERT_FALSE(HasVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetVectorInt64FeatureListSize(example));
  AddVectorInt64FeatureList(test_value[0], &example);
  ASSERT_THAT(GetVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_EQ(1, GetVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasVectorInt64FeatureList(example));
  AddVectorInt64FeatureList(test_value[1], &example);
  ASSERT_THAT(GetVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_THAT(GetVectorInt64FeatureListAt(example, 1),
              testing::ElementsAreArray(test_value[1]));
  ASSERT_EQ(test_value.size(), GetVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasVectorInt64FeatureList(example));
  ClearVectorInt64FeatureList(&example);
  ASSERT_FALSE(HasVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetVectorInt64FeatureListSize(example));
  ASSERT_EQ(GetVectorInt64FeatureListKey(), "vector_int64_feature_list");
}

TEST_F(MediaSequenceUtilTest, VectorFloatFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<float>> test_value = {{47.f, 42.f}, {3.f, 5.f}};

  ASSERT_FALSE(HasVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetVectorFloatFeatureListSize(example));
  AddVectorFloatFeatureList(test_value[0], &example);
  ASSERT_THAT(GetVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_EQ(1, GetVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasVectorFloatFeatureList(example));
  AddVectorFloatFeatureList(test_value[1], &example);
  ASSERT_THAT(GetVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value[0]));
  ASSERT_THAT(GetVectorFloatFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value[1]));
  ASSERT_EQ(test_value.size(), GetVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasVectorFloatFeatureList(example));
  ClearVectorFloatFeatureList(&example);
  ASSERT_FALSE(HasVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetVectorFloatFeatureListSize(example));
  ASSERT_EQ(GetVectorFloatFeatureListKey(), "vector_float_feature_list");
}

TEST_F(MediaSequenceUtilTest, FixedPrefixStringFeature) {
  tensorflow::SequenceExample example;
  std::string test_value_1 = "one";
  std::string test_value_2 = "two";

  ASSERT_FALSE(HasOneStringFeature(example));
  SetOneStringFeature(test_value_1, &example);
  ASSERT_TRUE(HasOneStringFeature(example));
  ASSERT_EQ(test_value_1, GetOneStringFeature(example));

  ASSERT_FALSE(HasTwoStringFeature(example));
  SetTwoStringFeature(test_value_2, &example);
  ASSERT_TRUE(HasTwoStringFeature(example));
  ASSERT_EQ(test_value_2, GetTwoStringFeature(example));

  ASSERT_EQ(test_value_1, GetOneStringFeature(example));
  ClearOneStringFeature(&example);
  ASSERT_FALSE(HasOneStringFeature(example));

  ClearOneStringFeature(&example);
  ASSERT_FALSE(HasOneStringFeature(example));

  ASSERT_EQ(GetOneStringFeatureKey(), "ONE/string_feature");
  ASSERT_EQ(GetTwoStringFeatureKey(), "TWO/string_feature");
}

TEST_F(MediaSequenceUtilTest, VariablePrefixStringFeature) {
  tensorflow::SequenceExample example;
  std::string prefix_1 = "ONE";
  std::string test_value_1 = "one";
  std::string prefix_2 = "TWO";
  std::string test_value_2 = "two";

  ASSERT_FALSE(HasStringFeature(prefix_1, example));
  SetStringFeature(prefix_1, test_value_1, &example);
  ASSERT_TRUE(HasStringFeature(prefix_1, example));
  ASSERT_EQ(test_value_1, GetStringFeature(prefix_1, example));

  ASSERT_FALSE(HasStringFeature(prefix_2, example));
  SetStringFeature(prefix_2, test_value_2, &example);
  ASSERT_TRUE(HasStringFeature(prefix_2, example));
  ASSERT_EQ(test_value_2, GetStringFeature(prefix_2, example));

  ASSERT_EQ(test_value_1, GetStringFeature(prefix_1, example));
  ClearStringFeature(prefix_2, &example);
  ASSERT_FALSE(HasStringFeature(prefix_2, example));

  ClearStringFeature(prefix_1, &example);
  ASSERT_FALSE(HasStringFeature(prefix_1, example));

  ASSERT_EQ(GetStringFeatureKey("ONE"), "ONE/string_feature");
  ASSERT_EQ(GetStringFeatureKey("TWO"), "TWO/string_feature");
}

TEST_F(MediaSequenceUtilTest, FixedPrefixInt64Feature) {
  tensorflow::SequenceExample example;
  int64 test_value_1 = 47;
  int64 test_value_2 = 49;

  ASSERT_FALSE(HasOneInt64Feature(example));
  SetOneInt64Feature(test_value_1, &example);
  ASSERT_TRUE(HasOneInt64Feature(example));
  ASSERT_EQ(test_value_1, GetOneInt64Feature(example));

  ASSERT_FALSE(HasTwoInt64Feature(example));
  SetTwoInt64Feature(test_value_2, &example);
  ASSERT_TRUE(HasTwoInt64Feature(example));
  ASSERT_EQ(test_value_2, GetTwoInt64Feature(example));

  ASSERT_EQ(test_value_1, GetOneInt64Feature(example));
  ClearOneInt64Feature(&example);
  ASSERT_FALSE(HasOneInt64Feature(example));

  ClearOneInt64Feature(&example);
  ASSERT_FALSE(HasOneInt64Feature(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixFloatFeature) {
  tensorflow::SequenceExample example;
  int64 test_value_1 = 47.0f;
  int64 test_value_2 = 49.0f;

  ASSERT_FALSE(HasOneFloatFeature(example));
  SetOneFloatFeature(test_value_1, &example);
  ASSERT_TRUE(HasOneFloatFeature(example));
  ASSERT_EQ(test_value_1, GetOneFloatFeature(example));

  ASSERT_FALSE(HasTwoFloatFeature(example));
  SetTwoFloatFeature(test_value_2, &example);
  ASSERT_TRUE(HasTwoFloatFeature(example));
  ASSERT_EQ(test_value_2, GetTwoFloatFeature(example));

  ASSERT_EQ(test_value_1, GetOneFloatFeature(example));
  ClearOneFloatFeature(&example);
  ASSERT_FALSE(HasOneFloatFeature(example));

  ClearOneFloatFeature(&example);
  ASSERT_FALSE(HasOneFloatFeature(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixStringVectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<std::string> test_value_1 = {"string1", "string2"};
  ::std::vector<std::string> test_value_2 = {"string3", "string4"};

  ASSERT_FALSE(HasOneStringVectorFeature(example));
  ASSERT_EQ(0, GetOneStringVectorFeatureSize(example));
  SetOneStringVectorFeature(test_value_1, &example);
  ASSERT_EQ(test_value_1.size(), GetOneStringVectorFeatureSize(example));
  ASSERT_TRUE(HasOneStringVectorFeature(example));
  ASSERT_THAT(GetOneStringVectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  AddOneStringVectorFeature(test_value_1[0], &example);
  AddOneStringVectorFeature(test_value_1[1], &example);
  ASSERT_EQ(test_value_1.size() * 2, GetOneStringVectorFeatureSize(example));

  ASSERT_FALSE(HasTwoStringVectorFeature(example));
  ASSERT_EQ(0, GetTwoStringVectorFeatureSize(example));
  SetTwoStringVectorFeature(test_value_2, &example);
  ASSERT_EQ(test_value_2.size(), GetTwoStringVectorFeatureSize(example));
  ASSERT_TRUE(HasTwoStringVectorFeature(example));
  ASSERT_THAT(GetTwoStringVectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  AddTwoStringVectorFeature(test_value_2[0], &example);
  AddTwoStringVectorFeature(test_value_2[1], &example);
  ASSERT_EQ(test_value_2.size() * 2, GetTwoStringVectorFeatureSize(example));
  ClearTwoStringVectorFeature(&example);
  ASSERT_FALSE(HasTwoStringVectorFeature(example));
  AddTwoStringVectorFeature(test_value_2[0], &example);
  AddTwoStringVectorFeature(test_value_2[1], &example);
  ASSERT_THAT(GetTwoStringVectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  ASSERT_EQ(test_value_2[1], GetTwoStringVectorFeatureAt(example, 1));
  SetTwoStringVectorFeature({"compile", "test"}, &example);

  ClearOneStringVectorFeature(&example);
  ASSERT_FALSE(HasOneStringVectorFeature(example));
  AddOneStringVectorFeature(test_value_1[0], &example);
  AddOneStringVectorFeature(test_value_1[1], &example);
  ASSERT_THAT(GetOneStringVectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  ASSERT_EQ(test_value_1[1], GetOneStringVectorFeatureAt(example, 1));
  SetOneStringVectorFeature({"compile", "test"}, &example);
}

TEST_F(MediaSequenceUtilTest, FixedPrefixInt64VectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<int64> test_value_1 = {47, 42};
  ::std::vector<int64> test_value_2 = {49, 47};

  ASSERT_FALSE(HasOneInt64VectorFeature(example));
  ASSERT_EQ(0, GetOneInt64VectorFeatureSize(example));
  SetOneInt64VectorFeature(test_value_1, &example);
  ASSERT_EQ(test_value_1.size(), GetOneInt64VectorFeatureSize(example));
  ASSERT_TRUE(HasOneInt64VectorFeature(example));
  ASSERT_THAT(GetOneInt64VectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  AddOneInt64VectorFeature(test_value_1[0], &example);
  AddOneInt64VectorFeature(test_value_1[1], &example);
  ASSERT_EQ(test_value_1.size() * 2, GetOneInt64VectorFeatureSize(example));

  ASSERT_FALSE(HasTwoInt64VectorFeature(example));
  ASSERT_EQ(0, GetTwoInt64VectorFeatureSize(example));
  SetTwoInt64VectorFeature(test_value_2, &example);
  ASSERT_EQ(test_value_2.size(), GetTwoInt64VectorFeatureSize(example));
  ASSERT_TRUE(HasTwoInt64VectorFeature(example));
  ASSERT_THAT(GetTwoInt64VectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  AddTwoInt64VectorFeature(test_value_2[0], &example);
  AddTwoInt64VectorFeature(test_value_2[1], &example);
  ASSERT_EQ(test_value_2.size() * 2, GetTwoInt64VectorFeatureSize(example));
  ClearTwoInt64VectorFeature(&example);
  ASSERT_FALSE(HasTwoInt64VectorFeature(example));
  AddTwoInt64VectorFeature(test_value_2[0], &example);
  AddTwoInt64VectorFeature(test_value_2[1], &example);
  ASSERT_THAT(GetTwoInt64VectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  ASSERT_EQ(test_value_2[1], GetTwoInt64VectorFeatureAt(example, 1));
  SetTwoInt64VectorFeature({3, 5}, &example);

  ClearOneInt64VectorFeature(&example);
  ASSERT_FALSE(HasOneInt64VectorFeature(example));
  AddOneInt64VectorFeature(test_value_1[0], &example);
  AddOneInt64VectorFeature(test_value_1[1], &example);
  ASSERT_THAT(GetOneInt64VectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  ASSERT_EQ(test_value_1[1], GetOneInt64VectorFeatureAt(example, 1));
  SetOneInt64VectorFeature({3, 5}, &example);
}

TEST_F(MediaSequenceUtilTest, FixedPrefixFloatVectorFeature) {
  tensorflow::SequenceExample example;
  ::std::vector<float> test_value_1 = {47.0f, 42.0f};
  ::std::vector<float> test_value_2 = {49.0f, 47.0f};

  ASSERT_FALSE(HasOneFloatVectorFeature(example));
  ASSERT_EQ(0, GetOneFloatVectorFeatureSize(example));
  SetOneFloatVectorFeature(test_value_1, &example);
  ASSERT_EQ(test_value_1.size(), GetOneFloatVectorFeatureSize(example));
  ASSERT_TRUE(HasOneFloatVectorFeature(example));
  ASSERT_THAT(GetOneFloatVectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  AddOneFloatVectorFeature(test_value_1[0], &example);
  AddOneFloatVectorFeature(test_value_1[1], &example);
  ASSERT_EQ(test_value_1.size() * 2, GetOneFloatVectorFeatureSize(example));

  ASSERT_FALSE(HasTwoFloatVectorFeature(example));
  ASSERT_EQ(0, GetTwoFloatVectorFeatureSize(example));
  SetTwoFloatVectorFeature(test_value_2, &example);
  ASSERT_EQ(test_value_2.size(), GetTwoFloatVectorFeatureSize(example));
  ASSERT_TRUE(HasTwoFloatVectorFeature(example));
  ASSERT_THAT(GetTwoFloatVectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  AddTwoFloatVectorFeature(test_value_2[0], &example);
  AddTwoFloatVectorFeature(test_value_2[1], &example);
  ASSERT_EQ(test_value_2.size() * 2, GetTwoFloatVectorFeatureSize(example));
  ClearTwoFloatVectorFeature(&example);
  ASSERT_FALSE(HasTwoFloatVectorFeature(example));
  AddTwoFloatVectorFeature(test_value_2[0], &example);
  AddTwoFloatVectorFeature(test_value_2[1], &example);
  ASSERT_THAT(GetTwoFloatVectorFeature(example),
              testing::ElementsAreArray(test_value_2));
  ASSERT_EQ(test_value_2[1], GetTwoFloatVectorFeatureAt(example, 1));
  SetTwoFloatVectorFeature({3.0f, 5.0f}, &example);

  ClearOneFloatVectorFeature(&example);
  ASSERT_FALSE(HasOneFloatVectorFeature(example));
  AddOneFloatVectorFeature(test_value_1[0], &example);
  AddOneFloatVectorFeature(test_value_1[1], &example);
  ASSERT_THAT(GetOneFloatVectorFeature(example),
              testing::ElementsAreArray(test_value_1));
  ASSERT_EQ(test_value_1[1], GetOneFloatVectorFeatureAt(example, 1));
  SetOneFloatVectorFeature({3.0f, 5.0f}, &example);
}

TEST_F(MediaSequenceUtilTest, FixedPrefixStringFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<std::string> test_value = {"string1", "string2"};

  ASSERT_FALSE(HasStringFeatureList(example));
  ASSERT_EQ(0, GetStringFeatureListSize(example));
  AddStringFeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetStringFeatureListAt(example, 0));
  ASSERT_EQ(1, GetStringFeatureListSize(example));
  ASSERT_TRUE(HasStringFeatureList(example));
  AddStringFeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetStringFeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetStringFeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetStringFeatureListSize(example));
  ASSERT_TRUE(HasStringFeatureList(example));
  ClearStringFeatureList(&example);
  ASSERT_FALSE(HasStringFeatureList(example));
  ASSERT_EQ(0, GetStringFeatureListSize(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixInt64FeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<int64> test_value = {47, 42};

  ASSERT_FALSE(HasInt64FeatureList(example));
  ASSERT_EQ(0, GetInt64FeatureListSize(example));
  AddInt64FeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetInt64FeatureListAt(example, 0));
  ASSERT_EQ(1, GetInt64FeatureListSize(example));
  ASSERT_TRUE(HasInt64FeatureList(example));
  AddInt64FeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetInt64FeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetInt64FeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetInt64FeatureListSize(example));
  ASSERT_TRUE(HasInt64FeatureList(example));
  ClearInt64FeatureList(&example);
  ASSERT_FALSE(HasInt64FeatureList(example));
  ASSERT_EQ(0, GetInt64FeatureListSize(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixFloatFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<float> test_value = {47.f, 42.f};

  ASSERT_FALSE(HasFloatFeatureList(example));
  ASSERT_EQ(0, GetFloatFeatureListSize(example));
  AddFloatFeatureList(test_value[0], &example);
  ASSERT_EQ(test_value[0], GetFloatFeatureListAt(example, 0));
  ASSERT_EQ(1, GetFloatFeatureListSize(example));
  ASSERT_TRUE(HasFloatFeatureList(example));
  AddFloatFeatureList(test_value[1], &example);
  ASSERT_EQ(test_value[0], GetFloatFeatureListAt(example, 0));
  ASSERT_EQ(test_value[1], GetFloatFeatureListAt(example, 1));
  ASSERT_EQ(test_value.size(), GetFloatFeatureListSize(example));
  ASSERT_TRUE(HasFloatFeatureList(example));
  ClearFloatFeatureList(&example);
  ASSERT_FALSE(HasFloatFeatureList(example));
  ASSERT_EQ(0, GetFloatFeatureListSize(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixVectorStringFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<std::string>> test_value_1 = {
      {"string1", "string2"}, {"string3", "string4"}};
  ::std::vector<::std::vector<std::string>> test_value_2 = {
      {"string5", "string6"}, {"string7", "string8"}};

  ASSERT_FALSE(HasOneVectorStringFeatureList(example));
  ASSERT_EQ(0, GetOneVectorStringFeatureListSize(example));
  AddOneVectorStringFeatureList(test_value_1[0], &example);
  ASSERT_THAT(GetOneVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_EQ(1, GetOneVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasOneVectorStringFeatureList(example));
  AddOneVectorStringFeatureList(test_value_1[1], &example);
  ASSERT_THAT(GetOneVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_THAT(GetOneVectorStringFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_1[1]));
  ASSERT_EQ(test_value_1.size(), GetOneVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasOneVectorStringFeatureList(example));

  ASSERT_FALSE(HasTwoVectorStringFeatureList(example));
  ASSERT_EQ(0, GetTwoVectorStringFeatureListSize(example));
  AddTwoVectorStringFeatureList(test_value_2[0], &example);
  ASSERT_THAT(GetTwoVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_EQ(1, GetTwoVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorStringFeatureList(example));
  AddTwoVectorStringFeatureList(test_value_2[1], &example);
  ASSERT_THAT(GetTwoVectorStringFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_THAT(GetTwoVectorStringFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_2[1]));
  ASSERT_EQ(test_value_2.size(), GetTwoVectorStringFeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorStringFeatureList(example));
  ClearTwoVectorStringFeatureList(&example);
  ASSERT_FALSE(HasTwoVectorStringFeatureList(example));
  ASSERT_EQ(0, GetTwoVectorStringFeatureListSize(example));

  ClearOneVectorStringFeatureList(&example);
  ASSERT_FALSE(HasOneVectorStringFeatureList(example));
  ASSERT_EQ(0, GetOneVectorStringFeatureListSize(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixVectorInt64FeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<int64>> test_value_1 = {{47, 42}, {3, 5}};
  ::std::vector<::std::vector<int64>> test_value_2 = {{49, 47}, {3, 5}};

  ASSERT_FALSE(HasOneVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetOneVectorInt64FeatureListSize(example));
  AddOneVectorInt64FeatureList(test_value_1[0], &example);
  ASSERT_THAT(GetOneVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_EQ(1, GetOneVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasOneVectorInt64FeatureList(example));
  AddOneVectorInt64FeatureList(test_value_1[1], &example);
  ASSERT_THAT(GetOneVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_THAT(GetOneVectorInt64FeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_1[1]));
  ASSERT_EQ(test_value_1.size(), GetOneVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasOneVectorInt64FeatureList(example));

  ASSERT_FALSE(HasTwoVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetTwoVectorInt64FeatureListSize(example));
  AddTwoVectorInt64FeatureList(test_value_2[0], &example);
  ASSERT_THAT(GetTwoVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_EQ(1, GetTwoVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorInt64FeatureList(example));
  AddTwoVectorInt64FeatureList(test_value_2[1], &example);
  ASSERT_THAT(GetTwoVectorInt64FeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_THAT(GetTwoVectorInt64FeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_2[1]));
  ASSERT_EQ(test_value_2.size(), GetTwoVectorInt64FeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorInt64FeatureList(example));
  ClearTwoVectorInt64FeatureList(&example);
  ASSERT_FALSE(HasTwoVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetTwoVectorInt64FeatureListSize(example));

  ClearOneVectorInt64FeatureList(&example);
  ASSERT_FALSE(HasOneVectorInt64FeatureList(example));
  ASSERT_EQ(0, GetOneVectorInt64FeatureListSize(example));
}

TEST_F(MediaSequenceUtilTest, FixedPrefixVectorFloatFeatureList) {
  tensorflow::SequenceExample example;
  ::std::vector<::std::vector<float>> test_value_1 = {{47.f, 42.f}, {3.f, 5.f}};
  ::std::vector<::std::vector<float>> test_value_2 = {{49.f, 47.f}, {3.f, 5.f}};

  ASSERT_FALSE(HasOneVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetOneVectorFloatFeatureListSize(example));
  AddOneVectorFloatFeatureList(test_value_1[0], &example);
  ASSERT_THAT(GetOneVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_EQ(1, GetOneVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasOneVectorFloatFeatureList(example));
  AddOneVectorFloatFeatureList(test_value_1[1], &example);
  ASSERT_THAT(GetOneVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_1[0]));
  ASSERT_THAT(GetOneVectorFloatFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_1[1]));
  ASSERT_EQ(test_value_1.size(), GetOneVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasOneVectorFloatFeatureList(example));

  ASSERT_FALSE(HasTwoVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetTwoVectorFloatFeatureListSize(example));
  AddTwoVectorFloatFeatureList(test_value_2[0], &example);
  ASSERT_THAT(GetTwoVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_EQ(1, GetTwoVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorFloatFeatureList(example));
  AddTwoVectorFloatFeatureList(test_value_2[1], &example);
  ASSERT_THAT(GetTwoVectorFloatFeatureListAt(example, 0),
              testing::ElementsAreArray(test_value_2[0]));
  ASSERT_THAT(GetTwoVectorFloatFeatureListAt(example, 1),
              testing::ElementsAreArray(test_value_2[1]));
  ASSERT_EQ(test_value_2.size(), GetTwoVectorFloatFeatureListSize(example));
  ASSERT_TRUE(HasTwoVectorFloatFeatureList(example));
  ClearTwoVectorFloatFeatureList(&example);
  ASSERT_FALSE(HasTwoVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetTwoVectorFloatFeatureListSize(example));

  ClearOneVectorFloatFeatureList(&example);
  ASSERT_FALSE(HasOneVectorFloatFeatureList(example));
  ASSERT_EQ(0, GetOneVectorFloatFeatureListSize(example));
}

}  // namespace
}  // namespace mediasequence
}  // namespace mediapipe
