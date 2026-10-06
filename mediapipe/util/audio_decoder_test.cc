// Copyright 2026 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/util/audio_decoder.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace {

TEST(AudioDecoderTest, DecodeS16LE) {
  // Create a dummy 16-bit PCM WAV file.
  const std::string test_file =
      absl::StrCat(testing::TempDir(), "/test_s16le.wav");

  // Write a simple WAV header and some dummy data.
  // 44 bytes header + 4 bytes data (2 samples, 1 channel, 16-bit)
  const uint8_t wav_data[] = {
      'R',  'I',  'F',  'F', 36 + 4, 0,   0,   0,  // ChunkSize
      'W',  'A',  'V',  'E', 'f',    'm', 't', ' ',
      16,   0,    0,    0,                         // Subchunk1Size
      1,    0,                                     // AudioFormat (PCM)
      1,    0,                                     // NumChannels (1)
      0x44, 0xac, 0,    0,                         // SampleRate (44100)
      0x88, 0x58, 0x01, 0,                         // ByteRate (44100 * 1 * 2)
      2,    0,                                     // BlockAlign (1 * 2)
      16,   0,                                     // BitsPerSample (16)
      'd',  'a',  't',  'a', 4,      0,   0,   0,  // Subchunk2Size
      0x00, 0x40,  // Sample 1: 16384 (0.5 in float)
      0x00, 0xc0,  // Sample 2: -16384 (-0.5 in float)
  };

  std::string wav_str(reinterpret_cast<const char*>(wav_data),
                      sizeof(wav_data));
  EXPECT_TRUE(file::SetContents(test_file, wav_str).ok());

  AudioDecoder decoder;
  mediapipe::AudioDecoderOptions options;
  options.add_audio_stream()->set_stream_index(0);
  EXPECT_TRUE(decoder.Initialize(test_file, options).ok());

  int options_index;
  Packet packet;
  EXPECT_TRUE(decoder.GetData(&options_index, &packet).ok());

  const Matrix& matrix = packet.Get<Matrix>();
  EXPECT_EQ(matrix.rows(), 1);
  EXPECT_EQ(matrix.cols(), 2);

  // 16384 / 32768.0 = 0.5
  EXPECT_NEAR(matrix(0, 0), 0.5f, 1e-5);
  // -16384 / 32768.0 = -0.5
  EXPECT_NEAR(matrix(0, 1), -0.5f, 1e-5);
}

TEST(AudioDecoderTest, DecodeS32LE) {
  // Create a dummy 32-bit PCM WAV file.
  const std::string test_file =
      absl::StrCat(testing::TempDir(), "/test_s32le.wav");

  // Write a simple WAV header and some dummy data.
  // 44 bytes header + 8 bytes data (2 samples, 1 channel, 32-bit)
  const uint8_t wav_data[] = {
      'R',  'I',  'F',  'F',  36 + 8, 0,   0,   0,  // ChunkSize
      'W',  'A',  'V',  'E',  'f',    'm', 't', ' ',
      16,   0,    0,    0,                          // Subchunk1Size
      1,    0,                                      // AudioFormat (PCM)
      1,    0,                                      // NumChannels (1)
      0x44, 0xac, 0,    0,                          // SampleRate (44100)
      0x10, 0xb1, 0x02, 0,                          // ByteRate (44100 * 1 * 4)
      4,    0,                                      // BlockAlign (1 * 4)
      32,   0,                                      // BitsPerSample (32)
      'd',  'a',  't',  'a',  8,      0,   0,   0,  // Subchunk2Size
      0x00, 0x00, 0x00, 0x40,  // Sample 1: 1073741824 (0.5 in float)
      0x00, 0x00, 0x00, 0xc0,  // Sample 2: -1073741824 (-0.5 in float)
  };

  std::string wav_str(reinterpret_cast<const char*>(wav_data),
                      sizeof(wav_data));
  EXPECT_TRUE(file::SetContents(test_file, wav_str).ok());

  AudioDecoder decoder;
  mediapipe::AudioDecoderOptions options;
  options.add_audio_stream()->set_stream_index(0);
  EXPECT_TRUE(decoder.Initialize(test_file, options).ok());

  int options_index;
  Packet packet;
  EXPECT_TRUE(decoder.GetData(&options_index, &packet).ok());

  const Matrix& matrix = packet.Get<Matrix>();
  EXPECT_EQ(matrix.rows(), 1);
  EXPECT_EQ(matrix.cols(), 2);

  // 1073741824 / 2147483648.0 = 0.5
  EXPECT_NEAR(matrix(0, 0), 0.5f, 1e-5);
  // -1073741824 / 2147483648.0 = -0.5
  EXPECT_NEAR(matrix(0, 1), -0.5f, 1e-5);
}

}  // namespace
}  // namespace mediapipe
