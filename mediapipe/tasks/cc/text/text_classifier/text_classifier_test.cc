/* Copyright 2022 The MediaPipe Authors. All Rights Reserved.

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

#include <cmath>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"
#include "mediapipe/tasks/cc/text/text_classifier/text_classifier_test_utils.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace text {
namespace text_classifier {
namespace {

using ::mediapipe::EqualsProto;
using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::kMediaPipeTasksPayload;
using ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr float kEpsilon = 0.001;
constexpr int kMaxSeqLen = 128;
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

class TextClassifierTest : public tflite_shims::testing::Test {};

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

}  // namespace
}  // namespace text_classifier
}  // namespace text
}  // namespace tasks
}  // namespace mediapipe
