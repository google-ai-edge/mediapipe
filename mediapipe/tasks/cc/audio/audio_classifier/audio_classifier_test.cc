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
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/tasks/cc/audio/core/running_mode.h"
#include "mediapipe/tasks/cc/audio/utils/test_utils.h"
#include "mediapipe/tasks/cc/common.h"
#include "mediapipe/tasks/cc/components/classifier_options.pb.h"
#include "mediapipe/tasks/cc/components/containers/category.pb.h"
#include "mediapipe/tasks/cc/components/containers/classifications.pb.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"

namespace mediapipe {
namespace tasks {
namespace audio {
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

void CheckSpeechClassificationResult(const ClassificationResult& result) {
  EXPECT_THAT(result.classifications_size(), testing::Eq(1));
  EXPECT_EQ(result.classifications(0).head_name(), "scores");
  EXPECT_EQ(result.classifications(0).head_index(), 0);
  EXPECT_THAT(result.classifications(0).entries_size(), testing::Eq(5));
  std::vector<int64> timestamps_ms = {0, 975, 1950, 2925};
  for (int i = 0; i < timestamps_ms.size(); i++) {
    EXPECT_THAT(result.classifications(0).entries(0).categories_size(),
                testing::Eq(521));
    const auto* top_category =
        &result.classifications(0).entries(0).categories(0);
    EXPECT_THAT(top_category->category_name(), testing::Eq("Speech"));
    EXPECT_GT(top_category->score(), 0.9f);
    EXPECT_EQ(result.classifications(0).entries(i).timestamp_ms(),
              timestamps_ms[i]);
  }
}

void CheckTwoHeadsClassificationResult(const ClassificationResult& result) {
  EXPECT_THAT(result.classifications_size(), testing::Eq(2));
  // Checks classification head #1.
  EXPECT_EQ(result.classifications(0).head_name(), "yamnet_classification");
  EXPECT_EQ(result.classifications(0).head_index(), 0);
  EXPECT_THAT(result.classifications(0).entries(0).categories_size(),
              testing::Eq(521));
  const auto* top_category =
      &result.classifications(0).entries(0).categories(0);
  EXPECT_THAT(top_category->category_name(),
              testing::Eq("Environmental noise"));
  EXPECT_GT(top_category->score(), 0.5f);
  EXPECT_EQ(result.classifications(0).entries(0).timestamp_ms(), 0);
  if (result.classifications(0).entries_size() == 2) {
    top_category = &result.classifications(0).entries(1).categories(0);
    EXPECT_THAT(top_category->category_name(), testing::Eq("Silence"));
    EXPECT_GT(top_category->score(), 0.99f);
    EXPECT_EQ(result.classifications(0).entries(1).timestamp_ms(), 975);
  }
  // Checks classification head #2.
  EXPECT_EQ(result.classifications(1).head_name(), "bird_classification");
  EXPECT_EQ(result.classifications(1).head_index(), 1);
  EXPECT_THAT(result.classifications(1).entries(0).categories_size(),
              testing::Eq(5));
  top_category = &result.classifications(1).entries(0).categories(0);
  EXPECT_THAT(top_category->category_name(),
              testing::Eq("Chestnut-crowned Antpitta"));
  EXPECT_GT(top_category->score(), 0.9f);
  EXPECT_EQ(result.classifications(1).entries(0).timestamp_ms(), 0);
}

ClassificationResult GenerateSpeechClassificationResult() {
  return ParseTextProtoOrDie<ClassificationResult>(
      R"pb(classifications {
             head_index: 0
             head_name: "scores"
             entries {
               categories { index: 0 score: 0.94140625 category_name: "Speech" }
               timestamp_ms: 0
             }
             entries {
               categories { index: 0 score: 0.9921875 category_name: "Speech" }
               timestamp_ms: 975
             }
             entries {
               categories { index: 0 score: 0.98828125 category_name: "Speech" }
               timestamp_ms: 1950
             }
             entries {
               categories { index: 0 score: 0.99609375 category_name: "Speech" }
               timestamp_ms: 2925
             }
             entries {
               # categories are filtered out due to the low scores.
               timestamp_ms: 3900
             }
           })pb");
}

void CheckStreamingModeClassificationResult(
    std::vector<ClassificationResult> outputs) {
  ASSERT_TRUE(outputs.size() == 5 || outputs.size() == 6);
  auto expected_results = GenerateSpeechClassificationResult();
  for (int i = 0; i < outputs.size() - 1; ++i) {
    EXPECT_THAT(outputs[i].classifications(0).entries(0),
                EqualsProto(expected_results.classifications(0).entries(i)));
  }
  int last_elem_index = outputs.size() - 1;
  EXPECT_EQ(
      mediapipe::Timestamp::Done().Value() / 1000,
      outputs[last_elem_index].classifications(0).entries(0).timestamp_ms());
}

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsForModelWithMetadata) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->classifier_options.max_results = 3;
  options->base_options.model_file_name =
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
                "'file_name' or 'file_descriptor_meta'."));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInvalidMaxResults) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->classifier_options.max_results = 0;
  options->base_options.model_file_name =
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
  options->base_options.model_file_name =
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
  options->base_options.model_file_name =
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
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  options->sample_rate = 16000;
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
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  options->result_callback =
      [](absl::StatusOr<ClassificationResult> status_or_result) {};
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

TEST_F(CreateFromOptionsTest, FailsWithMissingDefaultInputAudioSampleRate) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithoutMetadata);
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  options->result_callback =
      [](absl::StatusOr<ClassificationResult> status_or_result) {};
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::Create(std::move(options));

  EXPECT_EQ(audio_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_classifier_or.status().message(),
              HasSubstr("the sample rate must be specified"));
  EXPECT_THAT(audio_classifier_or.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

class ClassifyTest : public tflite_shims::testing::Test {};

TEST_F(ClassifyTest, Succeeds) {
  auto audio_buffer = GetAudioData(k16kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechClassificationResult(result);
}

TEST_F(ClassifyTest, SucceedsWithResampling) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechClassificationResult(result);
}

TEST_F(ClassifyTest, SucceedsWithInputsAtDifferentSampleRates) {
  auto audio_buffer_16k_hz = GetAudioData(k16kTestWavFilename);
  auto audio_buffer_48k_hz = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_16k_hz,
      audio_classifier->Classify(std::move(audio_buffer_16k_hz),
                                 /*audio_sample_rate=*/16000));
  CheckSpeechClassificationResult(result_16k_hz);
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_48k_hz,
      audio_classifier->Classify(std::move(audio_buffer_48k_hz),
                                 /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckSpeechClassificationResult(result_48k_hz);
}

TEST_F(ClassifyTest, SucceedsWithInsufficientData) {
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
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
  EXPECT_THAT(result.classifications_size(), testing::Eq(1));
  EXPECT_THAT(result.classifications(0).entries_size(), testing::Eq(1));
  EXPECT_THAT(result.classifications(0).entries(0).categories_size(),
              testing::Eq(521));
  EXPECT_THAT(
      result.classifications(0).entries(0).categories(0).category_name(),
      testing::Eq("Silence"));
  EXPECT_THAT(result.classifications(0).entries(0).categories(0).score(),
              testing::FloatEq(.800781f));
}

TEST_F(ClassifyTest, SucceedsWithMultiheadsModel) {
  auto audio_buffer = GetAudioData(k16kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsClassificationResult(result);
}

TEST_F(ClassifyTest, SucceedsWithMultiheadsModelAndResampling) {
  auto audio_buffer = GetAudioData(k44kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/44100));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsClassificationResult(result);
}

TEST_F(ClassifyTest,
       SucceedsWithMultiheadsModelAndInputsAtDifferentSampleRates) {
  auto audio_buffer_44k_hz = GetAudioData(k44kTestWavForTwoHeadsFilename);
  auto audio_buffer_16k_hz = GetAudioData(k16kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kTwoHeadsModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_44k_hz,
      audio_classifier->Classify(std::move(audio_buffer_44k_hz),
                                 /*audio_sample_rate=*/44100));
  CheckTwoHeadsClassificationResult(result_44k_hz);
  MP_ASSERT_OK_AND_ASSIGN(
      auto result_16k_hz,
      audio_classifier->Classify(std::move(audio_buffer_16k_hz),
                                 /*audio_sample_rate=*/16000));
  MP_ASSERT_OK(audio_classifier->Close());
  CheckTwoHeadsClassificationResult(result_16k_hz);
}

TEST_F(ClassifyTest, SucceedsWithMaxResultOption) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  options->classifier_options.score_threshold = 0.35f;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  EXPECT_THAT(result, EqualsProto(GenerateSpeechClassificationResult()));
}

TEST_F(ClassifyTest, SucceedsWithScoreThresholdOption) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.35f;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  EXPECT_THAT(result, EqualsProto(GenerateSpeechClassificationResult()));
}

TEST_F(ClassifyTest, SucceedsWithCategoryAllowlist) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.1f;
  options->classifier_options.category_allowlist.push_back("Speech");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  EXPECT_THAT(result, EqualsProto(GenerateSpeechClassificationResult()));
}

TEST_F(ClassifyTest, SucceedsWithCategoryDenylist) {
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.score_threshold = 0.9f;
  options->classifier_options.category_denylist.push_back("Speech");
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_classifier->Classify(std::move(audio_buffer),
                                              /*audio_sample_rate=*/48000));
  MP_ASSERT_OK(audio_classifier->Close());
  // All categroies with the "Speech" label are filtered out.
  EXPECT_THAT(result, EqualsProto(R"pb(classifications {
                                         head_index: 0
                                         head_name: "scores"
                                         entries { timestamp_ms: 0 }
                                         entries { timestamp_ms: 975 }
                                         entries { timestamp_ms: 1950 }
                                         entries { timestamp_ms: 2925 }
                                         entries { timestamp_ms: 3900 }
                                       })pb"));
}

class ClassifyAsyncTest : public tflite_shims::testing::Test {};

TEST_F(ClassifyAsyncTest, Succeeds) {
  constexpr int kSampleRateHz = 48000;
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  options->classifier_options.score_threshold = 0.3f;
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  options->sample_rate = kSampleRateHz;
  std::vector<ClassificationResult> outputs;
  options->result_callback =
      [&outputs](absl::StatusOr<ClassificationResult> status_or_result) {
        MP_ASSERT_OK_AND_ASSIGN(outputs.emplace_back(), status_or_result);
      };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioClassifier> audio_classifier,
                          AudioClassifier::Create(std::move(options)));
  int start_col = 0;
  while (start_col < audio_buffer.cols()) {
    int num_samples = std::min((int)(audio_buffer.cols() - start_col),
                               kYamnetNumOfAudioSamples * 3);
    MP_ASSERT_OK(audio_classifier->ClassifyAsync(
        audio_buffer.block(0, start_col, 1, num_samples),
        start_col * kMilliSecondsPerSecond / kSampleRateHz));
    start_col += kYamnetNumOfAudioSamples * 3;
  }
  MP_ASSERT_OK(audio_classifier->Close());
  CheckStreamingModeClassificationResult(outputs);
}

TEST_F(ClassifyAsyncTest, SucceedsWithNonDeterministicNumAudioSamples) {
  constexpr int kSampleRateHz = 48000;
  auto audio_buffer = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioClassifierOptions>();
  options->base_options.model_file_name =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->classifier_options.max_results = 1;
  options->classifier_options.score_threshold = 0.3f;
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  options->sample_rate = kSampleRateHz;
  std::vector<ClassificationResult> outputs;
  options->result_callback =
      [&outputs](absl::StatusOr<ClassificationResult> status_or_result) {
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
        audio_buffer.block(0, start_col, 1, num_samples),
        start_col * kMilliSecondsPerSecond / kSampleRateHz));
    start_col += num_samples;
  }
  MP_ASSERT_OK(audio_classifier->Close());
  CheckStreamingModeClassificationResult(outputs);
}

}  // namespace
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
