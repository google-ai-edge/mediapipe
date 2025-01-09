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

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/category.h"

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestBertModelPath[] = "bert_text_classifier.tflite";
constexpr char kTestString[] = "It's beautiful outside.";
constexpr float kPrecision = 1e-6;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(TextClassifierTest, SmokeTest) {
  std::string model_path = GetFullPath(kTestBertModelPath);
  TextClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ model_path.c_str()},
      /* classifier_options= */
      {/* display_names_locale= */ nullptr,
       /* max_results= */ -1,
       /* score_threshold= */ 0.0,
       /* category_allowlist= */ nullptr,
       /* category_allowlist_count= */ 0,
       /* category_denylist= */ nullptr,
       /* category_denylist_count= */ 0},
  };

  void* classifier = text_classifier_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(classifier, nullptr);

  TextClassifierResult result;
  text_classifier_classify(classifier, kTestString, &result,
                           /* error_msg */ nullptr);
  EXPECT_EQ(result.classifications_count, 1);
  EXPECT_EQ(result.classifications[0].categories_count, 2);
  EXPECT_EQ(std::string{result.classifications[0].categories[0].category_name},
            "positive");
  EXPECT_NEAR(result.classifications[0].categories[0].score, 0.999465,
              kPrecision);

  text_classifier_close_result(&result);
  text_classifier_close(classifier, /* error_msg */ nullptr);
}

TEST(TextClassifierTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  TextClassifierOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* classifier_options= */ {},
  };

  char* error_msg;
  void* classifier = text_classifier_create(&options, &error_msg);
  EXPECT_EQ(classifier, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("INVALID_ARGUMENT"));

  free(error_msg);
}

}  // namespace
