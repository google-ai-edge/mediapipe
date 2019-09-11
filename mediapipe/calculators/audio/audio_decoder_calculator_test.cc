// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

TEST(AudioDecoderCalculatorTest, TestWAV) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "AudioDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "AUDIO:audio"
        output_stream: "AUDIO_HEADER:audio_header"
        node_options {
          [type.googleapis.com/mediapipe.AudioDecoderOptions]: {
            audio_stream { stream_index: 0 }
          }
        })");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag("INPUT_FILE_PATH") = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/audio/"
                     "testdata/sine_wave_1k_44100_mono_2_sec_wav.audio"));
  MP_ASSERT_OK(runner.Run());
  MP_EXPECT_OK(runner.Outputs()
                   .Tag("AUDIO_HEADER")
                   .header.ValidateAsType<mediapipe::TimeSeriesHeader>());
  const mediapipe::TimeSeriesHeader& header =
      runner.Outputs()
          .Tag("AUDIO_HEADER")
          .header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(44100, header.sample_rate());
  EXPECT_EQ(1, header.num_channels());
  EXPECT_TRUE(runner.Outputs().Tag("AUDIO").packets.size() >=
              std::ceil(44100.0 * 2 / 2048));
}

TEST(AudioDecoderCalculatorTest, Test48KWAV) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "AudioDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "AUDIO:audio"
        output_stream: "AUDIO_HEADER:audio_header"
        node_options {
          [type.googleapis.com/mediapipe.AudioDecoderOptions]: {
            audio_stream { stream_index: 0 }
          }
        })");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag("INPUT_FILE_PATH") = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/audio/"
                     "testdata/sine_wave_1k_48000_stereo_2_sec_wav.audio"));
  MP_ASSERT_OK(runner.Run());
  MP_EXPECT_OK(runner.Outputs()
                   .Tag("AUDIO_HEADER")
                   .header.ValidateAsType<mediapipe::TimeSeriesHeader>());
  const mediapipe::TimeSeriesHeader& header =
      runner.Outputs()
          .Tag("AUDIO_HEADER")
          .header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(48000, header.sample_rate());
  EXPECT_EQ(2, header.num_channels());
  EXPECT_TRUE(runner.Outputs().Tag("AUDIO").packets.size() >=
              std::ceil(48000.0 * 2 / 1024));
}

TEST(AudioDecoderCalculatorTest, TestMP3) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "AudioDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "AUDIO:audio"
        output_stream: "AUDIO_HEADER:audio_header"
        node_options {
          [type.googleapis.com/mediapipe.AudioDecoderOptions]: {
            audio_stream { stream_index: 0 }
          }
        })");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag("INPUT_FILE_PATH") = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/audio/"
                     "testdata/sine_wave_1k_44100_stereo_2_sec_mp3.audio"));
  MP_ASSERT_OK(runner.Run());
  MP_EXPECT_OK(runner.Outputs()
                   .Tag("AUDIO_HEADER")
                   .header.ValidateAsType<mediapipe::TimeSeriesHeader>());
  const mediapipe::TimeSeriesHeader& header =
      runner.Outputs()
          .Tag("AUDIO_HEADER")
          .header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(44100, header.sample_rate());
  EXPECT_EQ(2, header.num_channels());
  EXPECT_TRUE(runner.Outputs().Tag("AUDIO").packets.size() >=
              std::ceil(44100.0 * 2 / 1152));
}

TEST(AudioDecoderCalculatorTest, TestAAC) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"(
        calculator: "AudioDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "AUDIO:audio"
        output_stream: "AUDIO_HEADER:audio_header"
        node_options {
          [type.googleapis.com/mediapipe.AudioDecoderOptions]: {
            audio_stream { stream_index: 0 }
          }
        })");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag("INPUT_FILE_PATH") = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/audio/"
                     "testdata/sine_wave_1k_44100_stereo_2_sec_aac.audio"));
  MP_ASSERT_OK(runner.Run());
  MP_EXPECT_OK(runner.Outputs()
                   .Tag("AUDIO_HEADER")
                   .header.ValidateAsType<mediapipe::TimeSeriesHeader>());
  const mediapipe::TimeSeriesHeader& header =
      runner.Outputs()
          .Tag("AUDIO_HEADER")
          .header.Get<mediapipe::TimeSeriesHeader>();
  EXPECT_EQ(44100, header.sample_rate());
  EXPECT_EQ(2, header.num_channels());
  EXPECT_TRUE(runner.Outputs().Tag("AUDIO").packets.size() >=
              std::ceil(44100.0 * 2 / 1024));
}

}  // namespace mediapipe
