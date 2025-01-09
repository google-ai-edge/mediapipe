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

#include "mediapipe/tasks/c/components/processors/classifier_options_converter.h"

#include <string>
#include <vector>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/processors/classifier_options.h"
#include "mediapipe/tasks/cc/components/processors/classifier_options.h"

namespace mediapipe::tasks::c::components::processors {

constexpr char kCategoryAllowlist[] = "fruit";
constexpr char kCategoryDenylist[] = "veggies";
constexpr char kDisplayNamesLocaleGerman[] = "de";

TEST(ClassifierOptionsConverterTest, ConvertsClassifierOptionsCustomValues) {
  std::vector<const char*> category_allowlist = {kCategoryAllowlist};
  std::vector<const char*> category_denylist = {kCategoryDenylist};

  ClassifierOptions c_classifier_options = {
      /* display_names_locale= */ kDisplayNamesLocaleGerman,
      /* max_results= */ 1,
      /* score_threshold= */ 0.1,
      /* category_allowlist= */ category_allowlist.data(),
      /* category_allowlist_count= */ 1,
      /* category_denylist= */ category_denylist.data(),
      /* category_denylist_count= */ 1};

  mediapipe::tasks::components::processors::ClassifierOptions
      cpp_classifier_options = {};

  CppConvertToClassifierOptions(c_classifier_options, &cpp_classifier_options);
  EXPECT_EQ(cpp_classifier_options.display_names_locale, "de");
  EXPECT_EQ(cpp_classifier_options.max_results, 1);
  EXPECT_FLOAT_EQ(cpp_classifier_options.score_threshold, 0.1);
  EXPECT_EQ(cpp_classifier_options.category_allowlist,
            std::vector<std::string>{"fruit"});
  EXPECT_EQ(cpp_classifier_options.category_denylist,
            std::vector<std::string>{"veggies"});
}

TEST(ClassifierOptionsConverterTest, ConvertsClassifierOptionsDefaultValues) {
  std::vector<const char*> category_allowlist = {kCategoryAllowlist};
  std::vector<const char*> category_denylist = {kCategoryDenylist};

  ClassifierOptions c_classifier_options = {/* display_names_locale= */ nullptr,
                                            /* max_results= */ -1,
                                            /* score_threshold= */ 0.0,
                                            /* category_allowlist= */ nullptr,
                                            /* category_allowlist_count= */ 0,
                                            /* category_denylist= */ nullptr,
                                            /* category_denylist_count= */ 0};

  mediapipe::tasks::components::processors::ClassifierOptions
      cpp_classifier_options = {};

  CppConvertToClassifierOptions(c_classifier_options, &cpp_classifier_options);
  EXPECT_EQ(cpp_classifier_options.display_names_locale, "en");
  EXPECT_EQ(cpp_classifier_options.max_results, -1);
  EXPECT_FLOAT_EQ(cpp_classifier_options.score_threshold, 0.0);
  EXPECT_EQ(cpp_classifier_options.category_allowlist,
            std::vector<std::string>{});
  EXPECT_EQ(cpp_classifier_options.category_denylist,
            std::vector<std::string>{});
}

}  // namespace mediapipe::tasks::c::components::processors
