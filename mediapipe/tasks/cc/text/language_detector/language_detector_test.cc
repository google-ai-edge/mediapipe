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

#include "mediapipe/tasks/cc/text/language_detector/language_detector.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe::tasks::text::language_detector {
namespace {

using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/text/";
constexpr char kInvalidModelPath[] = "i/do/not/exist.tflite";
constexpr char kLanguageDetector[] = "language_detector.tflite";

constexpr float kTolerance = 0.000001;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

absl::Status MatchesLanguageDetectorResult(
    const LanguageDetectorResult& expected,
    const LanguageDetectorResult& actual, float tolerance) {
  if (expected.size() != actual.size()) {
    return absl::FailedPreconditionError(absl::Substitute(
        "Expected $0 predictions, but got $1", expected.size(), actual.size()));
  }
  for (int i = 0; i < expected.size(); ++i) {
    if (expected[i].language_code != actual[i].language_code) {
      return absl::FailedPreconditionError(absl::Substitute(
          "Expected prediction $0 to have language_code $1, but got $2", i,
          expected[i].language_code, actual[i].language_code));
    }
    if (std::abs(expected[i].probability - actual[i].probability) > tolerance) {
      return absl::FailedPreconditionError(absl::Substitute(
          "Expected prediction $0 to have probability $1, but got $2", i,
          expected[i].probability, actual[i].probability));
    }
  }
  return absl::OkStatus();
}

}  // namespace

class LanguageDetectorTest : public tflite::testing::Test {};

TEST_F(LanguageDetectorTest, CreateFailsWithMissingModel) {
  auto options = std::make_unique<LanguageDetectorOptions>();
  options->base_options.model_asset_path = GetFullPath(kInvalidModelPath);
  absl::StatusOr<std::unique_ptr<LanguageDetector>> language_detector =
      LanguageDetector::Create(std::move(options));

  EXPECT_EQ(language_detector.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(language_detector.status().message(),
              HasSubstr("Unable to open file at"));
  EXPECT_THAT(language_detector.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(LanguageDetectorTest, TestL2CModel) {
  auto options = std::make_unique<LanguageDetectorOptions>();
  options->base_options.model_asset_path = GetFullPath(kLanguageDetector);
  options->classifier_options.score_threshold = 0.3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LanguageDetector> language_detector,
                          LanguageDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      LanguageDetectorResult result_en,
      language_detector->Detect("To be, or not to be, that is the question"));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "en", .probability = 0.999856}}, result_en,
      kTolerance));
  MP_ASSERT_OK_AND_ASSIGN(
      LanguageDetectorResult result_fr,
      language_detector->Detect(
          "Il y a beaucoup de bouches qui parlent et fort peu "
          "de têtes qui pensent."));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "fr", .probability = 0.999781}}, result_fr,
      kTolerance));
  MP_ASSERT_OK_AND_ASSIGN(
      LanguageDetectorResult result_ru,
      language_detector->Detect("это какой-то английский язык"));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "ru", .probability = 0.993362}}, result_ru,
      kTolerance));
}

TEST_F(LanguageDetectorTest, TestMultiplePredictions) {
  auto options = std::make_unique<LanguageDetectorOptions>();
  options->base_options.model_asset_path = GetFullPath(kLanguageDetector);
  options->classifier_options.score_threshold = 0.3;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LanguageDetector> language_detector,
                          LanguageDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(LanguageDetectorResult result_mixed,
                          language_detector->Detect("分久必合合久必分"));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "zh", .probability = 0.505424},
       {.language_code = "ja", .probability = 0.481617}},
      result_mixed, kTolerance));
}

TEST_F(LanguageDetectorTest, TestAllowList) {
  auto options = std::make_unique<LanguageDetectorOptions>();
  options->base_options.model_asset_path = GetFullPath(kLanguageDetector);
  options->classifier_options.category_allowlist = {"ja"};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LanguageDetector> language_detector,
                          LanguageDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(LanguageDetectorResult result_ja,
                          language_detector->Detect("分久必合合久必分"));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "ja", .probability = 0.481617}}, result_ja,
      kTolerance));
}

TEST_F(LanguageDetectorTest, TestDenyList) {
  auto options = std::make_unique<LanguageDetectorOptions>();
  options->base_options.model_asset_path = GetFullPath(kLanguageDetector);
  options->classifier_options.score_threshold = 0.3;
  options->classifier_options.category_denylist = {"ja"};
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LanguageDetector> language_detector,
                          LanguageDetector::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(LanguageDetectorResult result_zh,
                          language_detector->Detect("分久必合合久必分"));
  MP_EXPECT_OK(MatchesLanguageDetectorResult(
      {{.language_code = "zh", .probability = 0.505424}}, result_zh,
      kTolerance));
}

}  // namespace mediapipe::tasks::text::language_detector
