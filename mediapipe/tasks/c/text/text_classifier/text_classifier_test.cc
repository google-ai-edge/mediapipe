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
#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/c/test/test_utils.h"

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
  const std::string model_path = GetFullPath(kTestBertModelPath);
  TextClassifierOptions options = {
      .base_options = {.model_asset_path = model_path.c_str()},
      .classifier_options = {.max_results = -1, .score_threshold = 0.0},
  };

  MpTextClassifierPtr classifier;
  MP_ASSERT_OK(
      MpTextClassifierCreate(&options, &classifier, /*error_msg=*/nullptr));
  EXPECT_NE(classifier, nullptr);

  TextClassifierResult result;
  MP_ASSERT_OK(MpTextClassifierClassify(classifier, kTestString, &result,
                                        /*error_msg=*/nullptr));
  ASSERT_EQ(result.classifications_count, 1);
  ASSERT_EQ(result.classifications[0].categories_count, 2);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "positive");
  EXPECT_GE(result.classifications[0].categories[0].score, kScoreThreshold);

  MpTextClassifierCloseResult(&result);
  EXPECT_EQ(result.classifications, nullptr);
  MP_EXPECT_OK(MpTextClassifierClose(classifier, /*error_msg=*/nullptr));
}

TEST(TextClassifierTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  TextClassifierOptions options = {
      .base_options = {.model_asset_path = nullptr},
      .classifier_options = {},
  };

  MpTextClassifierPtr classifier;
  char* error_msg;
  MpStatus status = MpTextClassifierCreate(&options, &classifier, &error_msg);
  EXPECT_EQ(status, kMpInvalidArgument);

  EXPECT_THAT(error_msg,
              testing::HasSubstr("ExternalFile must specify at least one"));
  free(error_msg);
}

}  // namespace
