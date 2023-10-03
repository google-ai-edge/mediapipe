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

#include "mediapipe/tasks/c/components/containers/classification_result_converter.h"

#include <cstdlib>
#include <optional>
#include <string>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

namespace mediapipe::tasks::c::components::containers {

TEST(ClassificationResultConverterTest,
     ConvertsClassificationResulCustomCategory) {
  mediapipe::tasks::components::containers::ClassificationResult
      cpp_classification_result = {
          /* classifications= */ {{/* categories= */ {{
                                       /* index= */ 1,
                                       /* score= */ 0.1,
                                       /* category_name= */ std::nullopt,
                                       /* display_name= */ std::nullopt,
                                   }},
                                   /* head_index= */ 0,
                                   /* head_name= */ "foo"}},
          /* timestamp_ms= */ 42,
      };

  ClassificationResult c_classification_result;
  CppConvertToClassificationResult(cpp_classification_result,
                                   &c_classification_result);
  EXPECT_NE(c_classification_result.classifications, nullptr);
  EXPECT_EQ(c_classification_result.classifications_count, 1);
  EXPECT_NE(c_classification_result.classifications[0].categories, nullptr);
  EXPECT_EQ(c_classification_result.classifications[0].categories_count, 1);
  EXPECT_EQ(c_classification_result.classifications[0].head_index, 0);
  EXPECT_EQ(std::string(c_classification_result.classifications[0].head_name),
            "foo");
  EXPECT_EQ(c_classification_result.timestamp_ms, 42);
  EXPECT_EQ(c_classification_result.has_timestamp_ms, true);

  CppCloseClassificationResult(&c_classification_result);
}

TEST(ClassificationResultConverterTest,
     ConvertsClassificationResulEmptyCategory) {
  mediapipe::tasks::components::containers::ClassificationResult
      cpp_classification_result = {
          /* classifications= */ {{/* categories= */ {}, /* head_index= */ 0,
                                   /* head_name= */ std::nullopt}},
          /* timestamp_ms= */ std::nullopt,
      };

  ClassificationResult c_classification_result;
  CppConvertToClassificationResult(cpp_classification_result,
                                   &c_classification_result);
  EXPECT_NE(c_classification_result.classifications, nullptr);
  EXPECT_EQ(c_classification_result.classifications_count, 1);
  EXPECT_EQ(c_classification_result.classifications[0].categories, nullptr);
  EXPECT_EQ(c_classification_result.classifications[0].categories_count, 0);
  EXPECT_EQ(c_classification_result.classifications[0].head_index, 0);
  EXPECT_EQ(c_classification_result.classifications[0].head_name, nullptr);
  EXPECT_EQ(c_classification_result.timestamp_ms, 0);
  EXPECT_EQ(c_classification_result.has_timestamp_ms, false);

  CppCloseClassificationResult(&c_classification_result);
}

TEST(ClassificationResultConverterTest,
     ConvertsClassificationResultNoCategory) {
  mediapipe::tasks::components::containers::ClassificationResult
      cpp_classification_result = {
          /* classifications= */ {},
          /* timestamp_ms= */ std::nullopt,
      };

  ClassificationResult c_classification_result;
  CppConvertToClassificationResult(cpp_classification_result,
                                   &c_classification_result);
  EXPECT_EQ(c_classification_result.classifications, nullptr);
  EXPECT_EQ(c_classification_result.classifications_count, 0);
  EXPECT_EQ(c_classification_result.timestamp_ms, 0);
  EXPECT_EQ(c_classification_result.has_timestamp_ms, false);

  CppCloseClassificationResult(&c_classification_result);
}

TEST(ClassificationResultConverterTest, FreesMemory) {
  mediapipe::tasks::components::containers::ClassificationResult
      cpp_classification_result = {
          /* classifications= */ {{{}, 0, "foo"}},
          /* timestamp_ms= */ 42,
      };

  ClassificationResult c_classification_result;
  CppConvertToClassificationResult(cpp_classification_result,
                                   &c_classification_result);
  EXPECT_NE(c_classification_result.classifications, nullptr);

  CppCloseClassificationResult(&c_classification_result);
  EXPECT_EQ(c_classification_result.classifications, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
