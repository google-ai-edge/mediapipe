/* Copyright 2025 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUTHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/audio/audio_classifier/audio_classifier.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/c/components/containers/category.h"
#include "mediapipe/tasks/c/components/containers/classification_result.h"
#include "mediapipe/tasks/c/core/mp_status.h"
#include "mediapipe/tasks/cc/audio/utils/test_utils.h"

namespace {

using ::mediapipe::file::JoinPath;
using ::mediapipe::tasks::audio::internal::ReadWavFile;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/audio/";
constexpr char kTestModelPath[] =
    "yamnet_audio_classifier_with_metadata.tflite";
constexpr char kTestAudioClip[] = "speech_16000_hz_mono.wav";
constexpr double kTestSampleRate = 16000.0;
constexpr int kTestNumChannels = 1;

std::string GetFullPath(absl::string_view file_name) {
  return JoinPath("./", kTestDataDirectory, file_name);
}

void CheckSingleResult(const ClassificationResult& result, int index) {
  SCOPED_TRACE("Classification index: " + std::to_string(index));
  EXPECT_EQ(result.classifications_count, 1);
  EXPECT_EQ(result.classifications[0].head_index, 0);
  EXPECT_STREQ(result.classifications[0].head_name, "scores");
  EXPECT_EQ(result.classifications[0].categories_count, 1);
  EXPECT_EQ(result.classifications[0].categories[0].index, 0);
  EXPECT_STREQ(result.classifications[0].categories[0].category_name, "Speech");
  EXPECT_GT(result.classifications[0].categories[0].score, 0.9);
}

void CheckSpeechResult(MpAudioClassifierResult* classifier_result) {
  EXPECT_EQ(classifier_result->results_count, 5);
  // Ignore last result, which operates on a too small chunk to return relevant
  // results.
  std::vector<int64_t> timestamps_ms = {0, 975, 1950, 2925};
  for (int i = 0; i < timestamps_ms.size(); i++) {
    auto& result = classifier_result->results[i];
    EXPECT_EQ(result.timestamp_ms, timestamps_ms[i]);
    CheckSingleResult(result, i);
  }
}

// A struct to hold audio data and prevent the underlying buffer from being
// deallocated.
struct AudioData {
  std::unique_ptr<float[]> buffer;
  MpAudioData data;
};

AudioData LoadAudioData(const std::string& file_path) {
  int buffer_size;
  auto buffer = ReadWavFile(file_path, &buffer_size);

  AudioData audio_data;
  audio_data.buffer = std::move(*buffer);
  audio_data.data = {.num_channels = kTestNumChannels,
                     .sample_rate = kTestSampleRate,
                     .audio_data = audio_data.buffer.get(),
                     .audio_data_size = static_cast<size_t>(buffer_size)};
  return audio_data;
}

MpAudioClassifierOptions CreateAudioClassifierOptions(
    const char* model_path,
    MpAudioRunningMode running_mode = kMpAudioRunningModeAudioClips,
    MpAudioClassifierOptions::result_callback_fn result_callback = nullptr) {
  return {
      .base_options = {.model_asset_buffer = nullptr,
                       .model_asset_buffer_count = 0,
                       .model_asset_path = model_path},
      .classifier_options = {.display_names_locale = nullptr,
                             .max_results = 1,
                             .score_threshold = 0.0,
                             .category_allowlist = nullptr,
                             .category_allowlist_count = 0,
                             .category_denylist = nullptr,
                             .category_denylist_count = 0},
      .running_mode = running_mode,
      .result_callback = result_callback,
  };
}

TEST(AudioClassifierTest, ClassifyAudioClip) {
  const std::string model_path = GetFullPath(kTestModelPath);
  const std::string audio_clip_path = GetFullPath(kTestAudioClip);

  AudioData audio_data = LoadAudioData(audio_clip_path);
  MpAudioClassifierOptions options =
      CreateAudioClassifierOptions(model_path.c_str());

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  ASSERT_EQ(status, kMpOk);

  MpAudioClassifierResult result;
  status = MpAudioClassifierClassify(classifier, &audio_data.data, &result);
  ASSERT_EQ(status, kMpOk);

  CheckSpeechResult(&result);

  status = MpAudioClassifierCloseResult(&result);
  ASSERT_EQ(status, kMpOk);
  status = MpAudioClassifierClose(classifier);
  ASSERT_EQ(status, kMpOk);
}

absl::Notification* result_notification_ptr = nullptr;
void ResultCallback(MpStatus status, MpAudioClassifierResult* result) {
  if (result_notification_ptr == nullptr) {
    return;
  }
  ASSERT_EQ(status, kMpOk);
  EXPECT_EQ(result->results_count, 1);
  CheckSingleResult(result->results[0], /*index=*/0);
  result_notification_ptr->Notify();
  result_notification_ptr = nullptr;
}

TEST(AudioClassifierTest, ClassifyAudioStream) {
  absl::Notification notification;
  result_notification_ptr = &notification;

  const std::string model_path = GetFullPath(kTestModelPath);
  const std::string audio_clip_path = GetFullPath(kTestAudioClip);

  AudioData audio_data = LoadAudioData(audio_clip_path);
  MpAudioClassifierOptions options = CreateAudioClassifierOptions(
      model_path.c_str(), kMpAudioRunningModeAudioStream, ResultCallback);

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  ASSERT_EQ(status, kMpOk);

  status = MpAudioClassifierClassifyAsync(classifier, &audio_data.data, 0);
  ASSERT_EQ(status, kMpOk);

  notification.WaitForNotification();

  status = MpAudioClassifierClose(classifier);
  ASSERT_EQ(status, kMpOk);
}

TEST(AudioClassifierTest, CreateFailsWithUnnecessaryCallback) {
  const std::string model_path = GetFullPath(kTestModelPath);
  MpAudioClassifierOptions options = CreateAudioClassifierOptions(
      model_path.c_str(), kMpAudioRunningModeAudioClips, ResultCallback);

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  EXPECT_EQ(status, kMpInvalidArgument);
}

TEST(AudioClassifierTest, CreateFailsWithMissingCallback) {
  const std::string model_path = GetFullPath(kTestModelPath);
  MpAudioClassifierOptions options = CreateAudioClassifierOptions(
      model_path.c_str(), kMpAudioRunningModeAudioStream, nullptr);

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  EXPECT_NE(status, kMpOk);
}

TEST(AudioClassifierTest, ClassifyFailsWithWrongRunningMode) {
  const std::string model_path = GetFullPath(kTestModelPath);
  const std::string audio_clip_path = GetFullPath(kTestAudioClip);

  AudioData audio_data = LoadAudioData(audio_clip_path);
  MpAudioClassifierOptions options = CreateAudioClassifierOptions(
      model_path.c_str(), kMpAudioRunningModeAudioStream, ResultCallback);

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  ASSERT_EQ(status, kMpOk);

  MpAudioClassifierResult result;
  status = MpAudioClassifierClassify(classifier, &audio_data.data, &result);
  EXPECT_EQ(status, kMpInvalidArgument);

  status = MpAudioClassifierClose(classifier);
  ASSERT_EQ(status, kMpOk);
}

TEST(AudioClassifierTest, ClassifyAsyncFailsWithWrongRunningMode) {
  const std::string model_path = GetFullPath(kTestModelPath);
  const std::string audio_clip_path = GetFullPath(kTestAudioClip);

  AudioData audio_data = LoadAudioData(audio_clip_path);
  MpAudioClassifierOptions options =
      CreateAudioClassifierOptions(model_path.c_str());

  MpAudioClassifierPtr classifier;
  MpStatus status = MpAudioClassifierCreate(&options, &classifier);
  ASSERT_EQ(status, kMpOk);

  status = MpAudioClassifierClassifyAsync(classifier, &audio_data.data,
                                          /*timestamp_ms=*/0);
  EXPECT_EQ(status, kMpInvalidArgument);

  status = MpAudioClassifierClose(classifier);
  ASSERT_EQ(status, kMpOk);
}

}  // namespace
