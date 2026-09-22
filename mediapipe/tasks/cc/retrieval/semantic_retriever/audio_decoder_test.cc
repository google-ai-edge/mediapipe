/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

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

#include "mediapipe/tasks/cc/retrieval/semantic_retriever/audio_decoder.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/port/gmock.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/gtest.h"  // IWYU pragma: keep
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe::tasks::retrieval {
namespace {

using ::mediapipe::file::JoinPath;

TEST(AudioDecoderTest, TestDecodeAudioDataRealWavPCMMono) {
  std::string audio_path = JoinPath("./", "mediapipe/tasks/testdata/audio",
                                    "speech_16000_hz_mono.wav");

  MP_ASSERT_OK_AND_ASSIGN(auto data, AudioDecoder::DecodeAudioData(audio_path));
  ASSERT_FALSE(data.empty());
  for (float sample : data) {
    EXPECT_GE(sample, -1.0f);
    EXPECT_LE(sample, 1.0f);
  }
}

TEST(AudioDecoderTest, TestDecodeAudioDataStereoReturnsError) {
  // Write a valid 16-bit PCM Stereo (2 channels) WAV file
  std::string stereo_path = JoinPath(testing::TempDir(), "test_stereo.wav");
  std::ofstream file(stereo_path, std::ios::binary);

  file.write("RIFF", 4);
  uint32_t riff_size = 44;
  file.write(reinterpret_cast<const char*>(&riff_size), 4);
  file.write("WAVE", 4);
  file.write("fmt ", 4);
  uint32_t fmt_size = 16;
  file.write(reinterpret_cast<const char*>(&fmt_size), 4);
  uint16_t audio_format = 1;  // PCM
  file.write(reinterpret_cast<const char*>(&audio_format), 2);
  uint16_t num_channels = 2;  // Stereo!
  file.write(reinterpret_cast<const char*>(&num_channels), 2);
  uint32_t sample_rate = 16000;
  file.write(reinterpret_cast<const char*>(&sample_rate), 4);
  uint32_t byte_rate = 64000;
  file.write(reinterpret_cast<const char*>(&byte_rate), 4);
  uint16_t block_align = 4;
  file.write(reinterpret_cast<const char*>(&block_align), 2);
  uint16_t bits_per_sample = 16;
  file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
  file.write("data", 4);
  uint32_t data_size = 8;  // 2 frames * 2 channels * sizeof(int16_t)
  file.write(reinterpret_cast<const char*>(&data_size), 4);

  int16_t samples[] = {16384, 32767, -16384, -32768};
  file.write(reinterpret_cast<const char*>(samples), sizeof(samples));
  file.close();

  auto data_or = AudioDecoder::DecodeAudioData(stereo_path);
  EXPECT_EQ(data_or.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(data_or.status().message(),
              testing::HasSubstr("Only mono (1 channel) audio is supported"));
}

TEST(AudioDecoderTest, TestDecodeAudioBytesValidAndEmpty) {
  std::string audio_path = JoinPath("./", "mediapipe/tasks/testdata/audio",
                                    "speech_16000_hz_mono.wav");
  std::ifstream fs(audio_path, std::ios::binary);
  std::string wav_bytes((std::istreambuf_iterator<char>(fs)),
                        std::istreambuf_iterator<char>());

  uint32_t sample_rate = 0;
  MP_ASSERT_OK_AND_ASSIGN(
      auto data, AudioDecoder::DecodeAudioBytes(wav_bytes, &sample_rate));
  EXPECT_FALSE(data.empty());
  EXPECT_EQ(sample_rate, 16000u);

  auto empty_or = AudioDecoder::DecodeAudioBytes("");
  EXPECT_EQ(empty_or.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace mediapipe::tasks::retrieval
