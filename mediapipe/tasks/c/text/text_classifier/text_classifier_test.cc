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

#include "mediapipe/tasks/c/text/text_classifier/text_classifier.h"

#include <cassert>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/core/mp_status.h"

namespace {

using ::mediapipe::file::JoinPath;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestBertModelPath[] = "bert_text_classifier.tflite";
constexpr char kTestString[] = "It's beautiful outside.";
constexpr float kScoreThreshold = 0.95;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(TextClassifierTest, SmokeTest) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextClassifierOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .classifier_options = {.max_results = -1, .score_threshold = 0.0},
  };

  MpTextClassifierPtr classifier;
  EXPECT_EQ(MpTextClassifierCreate(&options, &classifier), kMpOk);
  EXPECT_NE(classifier, nullptr);

  TextClassifierResult result;
  EXPECT_EQ(MpTextClassifierClassify(classifier, kTestString, &result), kMpOk);
  ASSERT_EQ(result.classifications_count, 1);
  ASSERT_EQ(result.classifications[0].categories_count, 2);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "positive");
  EXPECT_GE(result.classifications[0].categories[0].score, kScoreThreshold);

  MpTextClassifierCloseResult(&result);
  EXPECT_EQ(result.classifications, nullptr);
  EXPECT_EQ(MpTextClassifierClose(classifier), kMpOk);
}

TEST(TextClassifierTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  TextClassifierOptions options = {
      .base_options = {.model_asset_path = nullptr},
      .classifier_options = {},
  };

  MpTextClassifierPtr classifier;
  EXPECT_EQ(MpTextClassifierCreate(&options, &classifier), kMpInvalidArgument);
}

}  // namespace
