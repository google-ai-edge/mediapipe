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

#include "mediapipe/tasks/cc/audio/audio_embedder/audio_embedder.h"

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
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "tensorflow/lite/test_util.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace audio_embedder {
namespace {

using ::absl::StatusOr;
using ::mediapipe::file::JoinPath;
using ::testing::HasSubstr;
using ::testing::Optional;

constexpr char kTestDataDirectory[] = "/mediapipe/tasks/testdata/audio";
constexpr char kModelWithMetadata[] = "yamnet_embedding_metadata.tflite";
constexpr char k16kTestWavFilename[] = "speech_16000_hz_mono.wav";
constexpr char k48kTestWavFilename[] = "speech_48000_hz_mono.wav";
constexpr char k16kTestWavForTwoHeadsFilename[] = "two_heads_16000_hz_mono.wav";
constexpr int kMilliSecondsPerSecond = 1000;
constexpr int kYamnetNumOfAudioSamples = 15600;
constexpr int kYamnetAudioSampleRate = 16000;

Matrix GetAudioData(absl::string_view filename) {
  std::string wav_file_path = JoinPath("./", kTestDataDirectory, filename);
  int buffer_size;
  auto audio_data = internal::ReadWavFile(wav_file_path, &buffer_size);
  Eigen::Map<Matrix> matrix_mapping(audio_data->get(), 1, buffer_size);
  return matrix_mapping.matrix();
}

class CreateFromOptionsTest : public tflite::testing::Test {};

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  auto audio_embedder =
      AudioEmbedder::Create(std::make_unique<AudioEmbedderOptions>());

  EXPECT_EQ(audio_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      audio_embedder.status().message(),
      HasSubstr("ExternalFile must specify at least one of 'file_content', "
                "'file_name', 'file_pointer_meta' or 'file_descriptor_meta'."));
  EXPECT_THAT(audio_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInitializationError))));
}

TEST_F(CreateFromOptionsTest, SucceedsForModelWithMetadata) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                          AudioEmbedder::Create(std::move(options)));
}

TEST_F(CreateFromOptionsTest, FailsWithIllegalCallbackInAudioClipsMode) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_CLIPS;
  options->result_callback = [](absl::StatusOr<AudioEmbedderResult>) {};

  auto audio_embedder = AudioEmbedder::Create(std::move(options));

  EXPECT_EQ(audio_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      audio_embedder.status().message(),
      HasSubstr("a user-defined result callback shouldn't be provided"));
  EXPECT_THAT(audio_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingCallbackInAudioStreamMode) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_STREAM;

  auto audio_embedder = AudioEmbedder::Create(std::move(options));

  EXPECT_EQ(audio_embedder.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(audio_embedder.status().message(),
              HasSubstr("a user-defined result callback must be provided"));
  EXPECT_THAT(audio_embedder.status().GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kInvalidTaskGraphConfigError))));
}

class EmbedTest : public tflite::testing::Test {};

TEST_F(EmbedTest, SucceedsWithSilentAudio) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_CLIPS;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                          AudioEmbedder::Create(std::move(options)));
  Matrix silent_data(1, kYamnetNumOfAudioSamples);
  silent_data.setZero();
  MP_ASSERT_OK_AND_ASSIGN(
      auto result, audio_embedder->Embed(silent_data, kYamnetAudioSampleRate));
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].embeddings[0].float_embedding.size(), 1024);
  constexpr float kValueDiffTolerance = 3e-6;
  EXPECT_NEAR(result[0].embeddings[0].float_embedding[0], 2.07613f,
              kValueDiffTolerance);
  EXPECT_NEAR(result[0].embeddings[0].float_embedding[1], 0.392721f,
              kValueDiffTolerance);
  EXPECT_NEAR(result[0].embeddings[0].float_embedding[2], 0.543622f,
              kValueDiffTolerance);
}

TEST_F(EmbedTest, SucceedsWithSameAudioAtDifferentSampleRates) {
  auto audio_buffer1 = GetAudioData(k16kTestWavFilename);
  auto audio_buffer2 = GetAudioData(k48kTestWavFilename);
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_CLIPS;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                          AudioEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(auto result1,
                          audio_embedder->Embed(audio_buffer1, 16000));
  MP_ASSERT_OK_AND_ASSIGN(auto result2,
                          audio_embedder->Embed(audio_buffer2, 48000));
  int expected_size = 5;
  ASSERT_EQ(result1.size(), expected_size);
  ASSERT_EQ(result2.size(), expected_size);
  MP_EXPECT_OK(audio_embedder->Close());
}

TEST_F(EmbedTest, SucceedsWithDifferentAudios) {
  auto audio_buffer1 = GetAudioData(k16kTestWavFilename);
  auto audio_buffer2 = GetAudioData(k16kTestWavForTwoHeadsFilename);
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_CLIPS;
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                          AudioEmbedder::Create(std::move(options)));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result1,
      audio_embedder->Embed(audio_buffer1, kYamnetAudioSampleRate));
  MP_ASSERT_OK_AND_ASSIGN(
      auto result2,
      audio_embedder->Embed(audio_buffer2, kYamnetAudioSampleRate));
  ASSERT_EQ(result1.size(), 5);
  ASSERT_EQ(result2.size(), 1);
  MP_EXPECT_OK(audio_embedder->Close());
}

class EmbedAsyncTest : public tflite::testing::Test {
 protected:
  void RunAudioEmbedderInStreamMode(std::string audio_file_name,
                                    int sample_rate_hz,
                                    std::vector<AudioEmbedderResult>* result) {
    auto audio_buffer = GetAudioData(audio_file_name);
    auto options = std::make_unique<AudioEmbedderOptions>();
    options->base_options.model_asset_path =
        JoinPath("./", kTestDataDirectory, kModelWithMetadata);
    options->running_mode = core::RunningMode::AUDIO_STREAM;
    options->result_callback =
        [result](absl::StatusOr<AudioEmbedderResult> status_or_result) {
          MP_ASSERT_OK_AND_ASSIGN(result->emplace_back(), status_or_result);
        };
    MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                            AudioEmbedder::Create(std::move(options)));
    int start_col = 0;
    static unsigned int rseed = 0;
    while (start_col < audio_buffer.cols()) {
      int num_samples = std::min(
          (int)(audio_buffer.cols() - start_col),
          rand_r(&rseed) % 10 + kYamnetNumOfAudioSamples * sample_rate_hz /
                                    kYamnetAudioSampleRate);
      MP_ASSERT_OK(audio_embedder->EmbedAsync(
          audio_buffer.block(0, start_col, 1, num_samples), sample_rate_hz,
          start_col * kMilliSecondsPerSecond / sample_rate_hz));
      start_col += num_samples;
    }
    MP_ASSERT_OK(audio_embedder->Close());
  }
};

TEST_F(EmbedAsyncTest, FailsWithOutOfOrderInputTimestamps) {
  auto options = std::make_unique<AudioEmbedderOptions>();
  options->base_options.model_asset_path =
      JoinPath("./", kTestDataDirectory, kModelWithMetadata);
  options->running_mode = core::RunningMode::AUDIO_STREAM;
  options->result_callback =
      [](absl::StatusOr<AudioEmbedderResult> status_or_result) { return; };
  MP_ASSERT_OK_AND_ASSIGN(std::unique_ptr<AudioEmbedder> audio_embedder,
                          AudioEmbedder::Create(std::move(options)));
  MP_ASSERT_OK(audio_embedder->EmbedAsync(Matrix(1, kYamnetNumOfAudioSamples),
                                          kYamnetAudioSampleRate, 100));
  auto status = audio_embedder->EmbedAsync(Matrix(1, kYamnetNumOfAudioSamples),
                                           kYamnetAudioSampleRate, 0);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              HasSubstr("timestamp must be monotonically increasing"));
  EXPECT_THAT(status.GetPayload(kMediaPipeTasksPayload),
              Optional(absl::Cord(absl::StrCat(
                  MediaPipeTasksStatus::kRunnerInvalidTimestampError))));
  MP_ASSERT_OK(audio_embedder->Close());
}

TEST_F(EmbedAsyncTest, SucceedsWithSameAudioAtDifferentSampleRates) {
  std::vector<AudioEmbedderResult> result1;
  RunAudioEmbedderInStreamMode(k16kTestWavFilename, 16000, &result1);
  std::vector<AudioEmbedderResult> result2;
  RunAudioEmbedderInStreamMode(k48kTestWavFilename, 48000, &result2);
  int expected_size = 5;
  ASSERT_EQ(result1.size(), expected_size);
  ASSERT_EQ(result2.size(), expected_size);
}

TEST_F(EmbedAsyncTest, SucceedsWithDifferentAudios) {
  std::vector<AudioEmbedderResult> result1;
  RunAudioEmbedderInStreamMode(k16kTestWavFilename, 16000, &result1);
  std::vector<AudioEmbedderResult> result2;
  RunAudioEmbedderInStreamMode(k16kTestWavForTwoHeadsFilename, 16000, &result2);
  ASSERT_EQ(result1.size(), 5);
  ASSERT_EQ(result2.size(), 1);
}

}  // namespace
}  // namespace audio_embedder
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe
