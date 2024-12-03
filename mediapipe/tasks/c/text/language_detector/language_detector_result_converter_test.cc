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

#include "mediapipe/tasks/c/text/language_detector/language_detector_result_converter.h"

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/text/language_detector/language_detector.h"
#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

namespace mediapipe::tasks::c::components::containers {

TEST(LanguageDetectorResultConverterTest,
     ConvertsLanguageDetectorResultCustomResult) {
  mediapipe::tasks::text::language_detector::LanguageDetectorResult
      cpp_detector_result = {{/* language_code= */ "fr",
                              /* probability= */ 0.5},
                             {/* language_code= */ "en",
                              /* probability= */ 0.5}};

  LanguageDetectorResult c_detector_result;
  CppConvertToLanguageDetectorResult(cpp_detector_result, &c_detector_result);
  EXPECT_NE(c_detector_result.predictions, nullptr);
  EXPECT_EQ(c_detector_result.predictions_count, 2);
  EXPECT_NE(c_detector_result.predictions[0].language_code, "fr");
  EXPECT_EQ(c_detector_result.predictions[0].probability, 0.5);

  CppCloseLanguageDetectorResult(&c_detector_result);
}

TEST(LanguageDetectorResultConverterTest, FreesMemory) {
  mediapipe::tasks::text::language_detector::LanguageDetectorResult
      cpp_detector_result = {{"fr", 0.5}};

  LanguageDetectorResult c_detector_result;
  CppConvertToLanguageDetectorResult(cpp_detector_result, &c_detector_result);
  EXPECT_NE(c_detector_result.predictions, nullptr);

  CppCloseLanguageDetectorResult(&c_detector_result);
  EXPECT_EQ(c_detector_result.predictions, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
