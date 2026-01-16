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

#include "mediapipe/tasks/cc/audio/audio_classifier/audio_classifier.h"

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/cc/audio/core/running_mode.h"
#include "mediapipe/tasks/cc/audio/utils/test_utils.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/containers/category.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace audio_classifier {
namespace {

using ::absl::StatusOr;
using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/audio";
constexpr char kModelWithMetadata[] =
    "yamnet_audio_classifier_with_metadata.tflite";
constexpr char kModelWithoutMetadata[] = "model_without_metadata.tflite";
constexpr char kTwoHeadsModelWithMetadata[] = "two_heads.tflite";
constexpr char k16kTestWavFilename[] = "speech_16000_hz_mono.wav";
constexpr char k48kTestWavFilename[] = "speech_48000_hz_mono.wav";
constexpr char k16kTestWavForTwoHeadsFilename[] = "two_heads_16000_hz_mono.wav";
constexpr char k44kTestWavForTwoHeadsFilename[] = "two_heads_44100_hz_mono.wav";
constexpr int kMilliSecondsPerSecond = 1000;
constexpr int kYamnetNumOfAudioSamples = 15600;

Matrix GetAudioData(absl::string_view filename) {
  std::string wav_file_path = JoinPath("./", kTestDataDirectory, filename);
  int buffer_size;
  auto audio_data = internal::ReadWavFile(wav_file_path, &buffer_size);
  Eigen::Map<Matrix> matrix_mapping(audio_data->get(), 1, buffer_size);
  return matrix_mapping.matrix();
}

// TODO: Compares the exact score values to capture unexpected
// changes in the inference pipeline.
void CheckSpeechResult(const std::vector<AudioClassifierResult>& result,
                       int expected_num_categories = 521) {
  EXPECT_EQ(result.size(), 5);
  // Ignore last result, which operates on a too small chunk to return relevant
  // results.
  std::vector<int64_t> timestamps_ms = {0, 975, 1950, 2925};
  for (int i = 0; i < timestamps_ms.size(); i++) {
    EXPECT_EQ(result[i].timestamp_ms, timestamps_ms[i]);
    EXPECT_EQ(result[i].classifications.size(), 1);
    auto classifications = result[i].classifications[0];
    EXPECT_EQ(classifications.head_index, 0);
    EXPECT_EQ(classifications.head_name, "scores");
    EXPECT_EQ(classifications.categories.size(), expected_num_categories);
    auto category = classifications.categories[0];
    EXPECT_EQ(category.index, 0);
    EXPECT_EQ(category.category_name, "Speech");
    EXPECT_GT(category.score, 0.9f);
  }
}

// TODO: Compares the exact score values to capture unexpected
// changes in the inference pipeline.
void CheckTwoHeadsResult(const std::vector<AudioClassifierResult>& result) {
  EXPECT_GE(result.size(), 1);
  EXPECT_LE(result.size(), 2);
  // Check the first result.
  EXPECT_EQ(result[0].timestamp_ms, 0);
  EXPECT_EQ(result[0].classifications.size(), 2);
  // Check the first head.
  EXPECT_EQ(result[0].classifications[0].head_index, 0);
  EXPECT_EQ(result[0].classifications[0].head_name, "yamnet_classification");
  EXPECT_EQ(result[0].classifications[0].categories.size(), 521);
  EXPECT_EQ(result[0].classifications[0].categories[0].index, 508);
  EXPECT_EQ(result[0].classifications[0].categories[0].category_name,
            "Environmental noise");
  EXPECT_GT(result[0].classifications[0].categories[0].score, 0.5f);
  // Check the second head.
  EXPECT_EQ(result[0].classifications[1].head_index, 1);
  EXPECT_EQ(result[0].classifications[1].head_name, "bird_classification");
  EXPECT_EQ(result[0].classifications[1].categories.size(), 5);
  EXPECT_EQ(result[0].classifications[1].categories[0].index, 4);
  EXPECT_EQ(result[0].classifications[1].categories[0].category_name,
            "Chestnut-crowned Antpitta");
  EXPECT_GT(result[0].classifications[1].categories[0].score, 0.93f);
  // Check the second result, if present.
  if (result.size() == 2) {
    EXPECT_EQ(result[1].timestamp_ms, 975);
    EXPECT_EQ(result[1].classifications.size(), 2);
    // Check the first head.
    EXPECT_EQ(result[1].classifications[0].head_index, 0);
    EXPECT_EQ(result[1].classifications[0].head_name, "yamnet_classification");
    EXPECT_EQ(result[1].classifications[0].categories.size(), 521);
    EXPECT_EQ(result[1].classifications[0].categories[0].index, 494);
    EXPECT_EQ(result[1].classifications[0].categories[0].category_name,
              "Silence");
    EXPECT_GT(result[1].classifications[0].categories[0].score, 0.99f);
    // Check the second head.
    EXPECT_EQ(result[1].classifications[1].head_index, 1);
    EXPECT_EQ(result[1].classifications[1].head_name, "bird_classification");
    EXPECT_EQ(result[1].classifications[1].categories.size(), 5);
    EXPECT_EQ(result[1].classifications[1].categories[0].index, 1);
    EXPECT_EQ(result[1].classifications[1].categories[0].category_name,
              "White-breasted Wood-Wren");
    EXPECT_GT(result[1].classifications[1].categories[0].score, 0.99f);
  }
}

void CheckStreamingModeResults(std::vector<AudioClassifierResult> outputs) {
  EXPECT_EQ(outputs.size(), 5);
  // Ignore last result, which operates on a too small chunk to return relevant
  // results.
  std::vector<int64_t> timestamps_ms = {0, 975, 1950, 2925};
  for (int i = 0; i < outputs.size() - 1; i++) {
    EXPECT_EQ(outputs[i].timestamp_ms.value(), timestamps_ms[i]);
    EXPECT_EQ(outputs[i].classifications.size(), 1);
    EXPECT_EQ(outputs[i].classifications[0].head_index, 0);
    EXPECT_EQ(outputs[i].classifications[0].head_name, "scores");
    EXPECT_EQ(outputs[i].classifications[0].categories.size(), 1);
    EXPECT_EQ(outputs[i].classifications[0].categories[0].index, 0);
    EXPECT_EQ(outputs[i].classifications[0].categories[0].category_name,
              "Speech");
    EXPECT_GT(outputs[i].classifications[0].categories[0].score, 0.9f);
  }
}

class CreateFromOptionsTest : public tflite::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsForModelWithMetadata) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->classifier_options.max_results = 3;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::make_unique<AudioClassifierOptions>());

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      audio_classifier_or.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInvalidMaxResults) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->classifier_options.max_results = 0;
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_classifier_or.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithCombinedAllowlistAndDenylist) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.category_allowlist.push_back("foo");
  options->classifier_options.category_denylist.push_back("bar");
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_classifier_or.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingMetadata) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_classifier_or.status().message(),
              HasSubstr("require TFLite Model Metadata"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingCallback) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_classifier_or.status().message(),
              HasSubstr("a user-defined result callback must be provided"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

TEST_F(CreateFromOptionsTest, FailsWithUnnecessaryCallback) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  options->result_callback =
      [](absl::StatusOr<AudioClassifierResult> status_or_result) {};
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      audio_classifier_or.status().message(),
      HasSubstr("a user-defined result callback shouldn't be provided"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

class ClassifyTest : public tflite::testing::Test {};

TEST_F(ClassifyTest, Succeeds) {
  auto audio_buffer = GetAudioData(k16kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result);
}

TEST_F(ClassifyTest, SucceedsWithResampling) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result);
}

TEST_F(ClassifyTest, SucceedsWithInputsAtDifferentSampleRates) {
  auto audio_buffer_16k_hz = GetAudioData(k16kTestWavFilename);
  auto audio_buffer_48k_hz = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_16k_hz,
      audio_classifier->Classify(std::move(audio_buffer_16k_hz),
                                 /*audio_sample_rate=*/16000));
  CheckSpeechResult(result_16k_hz);
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_48k_hz,
      audio_classifier->Classify(std::move(audio_buffer_48k_hz),
                                 /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result_48k_hz);
}

TEST_F(ClassifyTest, SucceedsWithInsufficientData) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  // The input audio buffer doesn't have sufficient data (15600 samples).
  // Expects that the audio classifier will append zero-paddings.
  Matrix zero_matrix(1, 14000);
  zero_matrix.setZero();
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(zero_matrix), 16000));
  MP_ASSERT_OK(audio_classifier->Close());
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].timestamp_ms, 0);
  EXPECT_EQ(result[0].classifications.size(), 1);
  EXPECT_EQ(result[0].classifications[0].head_index, 0);
  EXPECT_EQ(result[0].classifications[0].head_name, "scores");
  EXPECT_EQ(result[0].classifications[0].categories.size(), 521);
  EXPECT_EQ(result[0].classifications[0].categories[0].index, 494);
  EXPECT_EQ(result[0].classifications[0].categories[0].category_name,
            "Silence");
  EXPECT_FLOAT_EQ(result[0].classifications[0].categories[0].score, 0.800781f);
}

TEST_F(ClassifyTest, SucceedsWithMultiheadsModel) {
  auto audio_buffer = GetAudioData(k16kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsResult(result);
}

TEST_F(ClassifyTest, SucceedsWithMultiheadsModelAndResampling) {
  auto audio_buffer = GetAudioData(k44kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/44100));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsResult(result);
}

TEST_F(ClassifyTest,
       SucceedsWithMultiheadsModelAndInputsAtDifferentSampleRates) {
  auto audio_buffer_44k_hz = GetAudioData(k44kTestWavForTwoHeadsFilename);
  auto audio_buffer_16k_hz = GetAudioData(k16kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_44k_hz,
      audio_classifier->Classify(std::move(audio_buffer_44k_hz),
                                 /*audio_sample_rate=*/44100));
  CheckTwoHeadsResult(result_44k_hz);
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_16k_hz,
      audio_classifier->Classify(std::move(audio_buffer_16k_hz),
                                 /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsResult(result_16k_hz);
}

TEST_F(ClassifyTest, SucceedsWithMaxResultOption) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result, /*expected_num_categories=*/1);
}

TEST_F(ClassifyTest, SucceedsWithScoreThresholdOption) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.35f;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result, /*expected_num_categories=*/1);
}

TEST_F(ClassifyTest, SucceedsWithCategoryAllowlist) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.1f;
  options->classifier_options.category_allowlist.push_back("Speech");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechResult(result, /*expected_num_categories=*/1);
}

TEST_F(ClassifyTest, SucceedsWithCategoryDenylist) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.9f;
  options->classifier_options.category_denylist.push_back("Speech");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  // All categories with the "Speech" label are filtered out.
  std::vector<int64_t> timestamps_ms = {0, 975, 1950, 2925};
  for (int i = 0; i < timestamps_ms.size(); i++) {
    EXPECT_EQ(result[i].timestamp_ms, timestamps_ms[i]);
    EXPECT_EQ(result[i].classifications.size(), 1);
    auto classifications = result[i].classifications[0];
    EXPECT_EQ(classifications.head_index, 0);
    EXPECT_EQ(classifications.head_name, "scores");
    EXPECT_TRUE(classifications.categories.empty());
  }
}

class ClassifyAsyncTest : public tflite::testing::Test {};

TEST_F(ClassifyAsyncTest, Succeeds) {
  constexpr int kSampleRateHz = 48000;
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  options->classifier_options.score_threshold = 0.3f;
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  std::vector<AudioClassifierResult> outputs;
  options->result_callback =
      [&outputs](absl::StatusOr<AudioClassifierResult> status_or_result) {
        MP_ASSERT_OK_AND_ASSIGN(outputs.emplace_back(), status_or_result);
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  int start_col = 0;
  while (start_col < audio_buffer.cols()) {
    int num_samples = std::min((int)(audio_buffer.cols() - start_col),
                               kYamnetNumOfAudioSamples * 3);
    MP_ASSERT_OK(audio_classifier->ClassifyAsync(
        audio_buffer.block(0, start_col, 1, num_samples), kSampleRateHz,
        start_col * kMilliSecondsPerSecond / kSampleRateHz));
    start_col += kYamnetNumOfAudioSamples * 3;
  }
  MP_ASSERT_OK(audio_classifier->Close());
  CheckStreamingModeResults(outputs);
}

TEST_F(ClassifyAsyncTest, SucceedsWithNonDeterministicNumAudioSamples) {
  constexpr int kSampleRateHz = 48000;
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  options->classifier_options.score_threshold = 0.3f;
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  std::vector<AudioClassifierResult> outputs;
  options->result_callback =
      [&outputs](absl::StatusOr<AudioClassifierResult> status_or_result) {
        MP_ASSERT_OK_AND_ASSIGN(outputs.emplace_back(), status_or_result);
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  int start_col = 0;
  static unsigned int rseed = 0;
  while (start_col < audio_buffer.cols()) {
    int num_samples =
        std::min((int)(audio_buffer.cols() - start_col),
                 rand_r(&rseed) % 10 + kYamnetNumOfAudioSamples * 3);
    MP_ASSERT_OK(audio_classifier->ClassifyAsync(
        audio_buffer.block(0, start_col, 1, num_samples), kSampleRateHz,
        start_col * kMilliSecondsPerSecond / kSampleRateHz));
    start_col += num_samples;
  }
  MP_ASSERT_OK(audio_classifier->Close());
  CheckStreamingModeResults(outputs);
}

}  // namespace
}  // namespace audio_classifier
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
