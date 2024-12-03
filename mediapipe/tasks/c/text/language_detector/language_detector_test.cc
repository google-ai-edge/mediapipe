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

#include "mediapipe/tasks/c/text/language_detector/language_detector.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace {

using ::mediapipe::file::JoinPath;
using testing::HasSubstr;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kTestLanguageDetectorModelPath[] = "language_detector.tflite";
constexpr char kTestString[] =
    "Il y a beaucoup de bouches qui parlent et fort peu "
    "de têtes qui pensent.";
constexpr float kPrecision = 1e-6;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

TEST(LanguageDetectorTest, SmokeTest) {
  std::string model_path = GetFullPath(kTestLanguageDetectorModelPath);
  LanguageDetectorOptions options = {
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

  void* detector = language_detector_create(&options, /* error_msg */ nullptr);
  EXPECT_NE(detector, nullptr);

  LanguageDetectorResult result;
  language_detector_detect(detector, kTestString, &result,
                           /* error_msg */ nullptr);
  EXPECT_EQ(std::string(result.predictions[0].language_code), "fr");
  EXPECT_NEAR(result.predictions[0].probability, 0.999781, kPrecision);

  language_detector_close_result(&result);
  language_detector_close(detector, /* error_msg */ nullptr);
}

TEST(LanguageDetectorTest, ErrorHandling) {
  // It is an error to set neither the asset buffer nor the path.
  LanguageDetectorOptions options = {
      /* base_options= */ {/* model_asset_buffer= */ nullptr,
                           /* model_asset_buffer_count= */ 0,
                           /* model_asset_path= */ nullptr},
      /* classifier_options= */ {},
  };

  char* error_msg;
  void* detector = language_detector_create(&options, &error_msg);
  EXPECT_EQ(detector, nullptr);

  EXPECT_THAT(error_msg, HasSubstr("INVALID_ARGUMENT"));

  free(error_msg);
}

}  // namespace
