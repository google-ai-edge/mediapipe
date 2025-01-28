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

#include "mediapipe/tasks/cc/text/text_classifier/text_classifier.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier_test_utils.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe::tasks::text::text_classifier {
namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::kMediaPipeTasksPayload;
using ::mediapipe::tasks::components::containers::Category;
using ::mediapipe::tasks::components::containers::Classifications;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr int kMaxSeqLen = 128;
const float kPrecision = 1e-5;
constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestBertModelPath[] = "bert_text_classifier.tflite";
constexpr char kInvalidModelPath[] = "i/do/not/exist.tflite";
constexpr char kTestRegexModelPath[] =
    "test_model_text_classifier_with_regex_tokenizer.tflite";
constexpr char kStringToBoolModelPath[] =
    "test_model_text_classifier_bool_output.tflite";

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

// Checks that the two provided `TextClassifierResult` are equal, with a
// tolerancy on floating-point score to account for numerical instabilities.
// TODO: create shared matcher for ClassificationResult.
void ExpectApproximatelyEqual(const TextClassifierResult& actual,
                              const TextClassifierResult& expected) {
  ASSERT_EQ(actual.classifications.size(), expected.classifications.size());
  for (int i = 0; i < actual.classifications.size(); ++i) {
    const Classifications& a = actual.classifications[i];
    const Classifications& b = expected.classifications[i];
    EXPECT_EQ(a.head_index, b.head_index);
    EXPECT_EQ(a.head_name, b.head_name);
    EXPECT_EQ(a.categories.size(), b.categories.size());
    for (int j = 0; j < a.categories.size(); ++j) {
      const Category& x = a.categories[j];
      const Category& y = b.categories[j];
      EXPECT_EQ(x.index, y.index);
      EXPECT_EQ(x.category_name, y.category_name);
      EXPECT_EQ(x.display_name, y.display_name);
    }
  }
}

}  // namespace

class TextClassifierTest : public tflite::testing::Test {};

TEST_F(TextClassifierTest, CreateSucceedsWithBertModel) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kTestBertModelPath);
  MP_ASSERT_OK(TextClassifier::Create(std::move(options)));
}

TEST_F(TextClassifierTest, CreateFailsWithMissingBaseOptions) {
  auto options = std::make_unique<TextClassifierOptions>();
  StatusOr<std::unique_ptr<TextClassifier>> classifier =
      TextClassifier::Create(std::move(options));

  EXPECT_EQ(classifier.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      classifier.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(TextClassifierTest, CreateFailsWithMissingModel) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kInvalidModelPath);
  StatusOr<std::unique_ptr<TextClassifier>> classifier =
      TextClassifier::Create(std::move(options));

  EXPECT_EQ(classifier.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(classifier.status().message(),
              HasSubstr("Unable to open file at"));
  EXPECT_THAT(classifier.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(TextClassifierTest, CreateSucceedsWithRegexModel) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kTestRegexModelPath);
  MP_ASSERT_OK(TextClassifier::Create(std::move(options)));
}

TEST_F(TextClassifierTest, TextClassifierWithBert) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kTestBertModelPath);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextClassifier> classifier,
                          TextClassifier::Create(std::move(options)));

  TextClassifierResult negative_expected;
  TextClassifierResult positive_expected;

#ifdef _WIN32
  negative_expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/0, /*score=*/0.956124, /*category_name=*/"negative"},
          {/*index=*/1, /*score=*/0.043875, /*category_name=*/"positive"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  positive_expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/1, /*score=*/0.999951, /*category_name=*/"positive"},
          {/*index=*/0, /*score=*/0.000048, /*category_name=*/"negative"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
#else
  negative_expected.classifications.emplace_back(Classifications{
      /*categories=*/{{0, 0.963325, "negative"}, {1, 0.036674, "positive"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  positive_expected.classifications.emplace_back(Classifications{
      /*categories=*/{{1, 0.9999370, "positive"}, {0, 0.0000629, "negative"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
#endif  // _WIN32

  MP_ASSERT_OK_AND_ASSIGN(
      TextClassifierResult negative_result,
      classifier->Classify("unflinchingly bleak and desperate"));
  ExpectApproximatelyEqual(negative_result, negative_expected);

  MP_ASSERT_OK_AND_ASSIGN(
      TextClassifierResult positive_result,
      classifier->Classify("it's a charming and often affecting journey"));
  ExpectApproximatelyEqual(positive_result, positive_expected);

  MP_ASSERT_OK(classifier->Close());
}

TEST_F(TextClassifierTest, TextClassifierWithIntInputs) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kTestRegexModelPath);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextClassifier> classifier,
                          TextClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(TextClassifierResult negative_result,
                          classifier->Classify("What a waste of my time."));
  TextClassifierResult negative_expected;
  negative_expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/0, /*score=*/0.813130, /*category_name=*/"Negative"},
          {/*index=*/1, /*score=*/0.186870, /*category_name=*/"Positive"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(negative_result, negative_expected);

  MP_ASSERT_OK_AND_ASSIGN(
      TextClassifierResult positive_result,
      classifier->Classify("This is the best movie I’ve seen in recent years."
                           "Strongly recommend it!"));
  TextClassifierResult positive_expected;
  positive_expected.classifications.emplace_back(Classifications{
      /*categories=*/{
          {/*index=*/1, /*score=*/0.513427, /*category_name=*/"Positive"},
          {/*index=*/0, /*score=*/0.486573, /*category_name=*/"Negative"}},
      /*head_index=*/0,
      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(positive_result, positive_expected);

  MP_ASSERT_OK(classifier->Close());
}

TEST_F(TextClassifierTest, TextClassifierWithStringToBool) {
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kStringToBoolModelPath);
  options->base_options.op_resolver = CreateCustomResolver();
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextClassifier> classifier,
                          TextClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(TextClassifierResult result,
                          classifier->Classify("hello"));

  // Binary outputs causes flaky ordering, so we compare manually.
  ASSERT_EQ(result.classifications.size(), 1);
  ASSERT_EQ(result.classifications[0].head_index, 0);
  ASSERT_EQ(result.classifications[0].categories.size(), 3);
  ASSERT_EQ(result.classifications[0].categories[0].score, 1);
  ASSERT_LT(result.classifications[0].categories[0].index, 2);  // i.e O or 1.
  ASSERT_EQ(result.classifications[0].categories[1].score, 1);
  ASSERT_LT(result.classifications[0].categories[1].index, 2);  // i.e 0 or 1.
  ASSERT_EQ(result.classifications[0].categories[2].score, 0);
  ASSERT_EQ(result.classifications[0].categories[2].index, 2);
  MP_ASSERT_OK(classifier->Close());
}

TEST_F(TextClassifierTest, BertLongPositive) {
  std::stringstream ss_for_positive_review;
  ss_for_positive_review
      << "it's a charming and often affecting journey and this is a long";
  for (int i = 0; i < kMaxSeqLen; ++i) {
    ss_for_positive_review << " long";
  }
  ss_for_positive_review << " movie review";
  auto options = std::make_unique<TextClassifierOptions>();
  options->base_options.model_asset_path = GetFullPath(kTestBertModelPath);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<TextClassifier> classifier,
                          TextClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(TextClassifierResult result,
                          classifier->Classify(ss_for_positive_review.str()));
  TextClassifierResult expected;
  std::vector<Category> categories;

// Predicted scores are slightly different on Windows.
#ifdef _WIN32
  categories.push_back(
      {/*index=*/1, /*score=*/0.976686, /*category_name=*/"positive"});
  categories.push_back(
      {/*index=*/0, /*score=*/0.023313, /*category_name=*/"negative"});
#else
  categories.push_back({1, 0.981097, "positive"});
  categories.push_back({0, 0.018902, "negative"});
#endif  // _WIN32

  expected.classifications.emplace_back(
      Classifications{/*categories=*/categories,
                      /*head_index=*/0,
                      /*head_name=*/"probability"});
  ExpectApproximatelyEqual(result, expected);
  MP_ASSERT_OK(classifier->Close());
}

}  // namespace mediapipe::tasks::text::text_classifier
