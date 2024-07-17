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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/test_util.h"

namespace mediapipe {

namespace {

constexpr char kVideoTag[] = "VIDEO";
constexpr char kVideoPrestreamTag[] = "VIDEO_PRESTREAM";
constexpr char kInputFilePathTag[] = "INPUT_FILE_PATH";
constexpr char kTestPackageRoot[] = "mediapipe/calculators/video";

TEST(OpenCvVideoDecoderCalculatorTest, TestMp4Avc720pVideo) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "OpenCvVideoDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "VIDEO:video"
        output_stream: "VIDEO_PRESTREAM:video_prestream")pb");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag(kInputFilePathTag) =
      MakePacket<std::string>(file::JoinPath(GetTestDataDir(kTestPackageRoot),
                                             "format_MP4_AVC720P_AAC.video"));
  MP_EXPECT_OK(runner.Run());

  EXPECT_EQ(runner.Outputs().Tag(kVideoPrestreamTag).packets.size(), 1);
  MP_EXPECT_OK(runner.Outputs()
                   .Tag(kVideoPrestreamTag)
                   .packets[0]
                   .ValidateAsType<VideoHeader>());
  const mediapipe::VideoHeader& header =
      runner.Outputs().Tag(kVideoPrestreamTag).packets[0].Get<VideoHeader>();
  EXPECT_EQ(ImageFormat::SRGB, header.format);
  EXPECT_EQ(1280, header.width);
  EXPECT_EQ(640, header.height);
  EXPECT_FLOAT_EQ(6.0f, header.duration);
  EXPECT_FLOAT_EQ(30.0f, header.frame_rate);
  // The number of the output packets should be 180.
  int num_of_packets = runner.Outputs().Tag(kVideoTag).packets.size();
  EXPECT_GE(num_of_packets, 180);
  for (int i = 0; i < num_of_packets; ++i) {
    Packet image_frame_packet = runner.Outputs().Tag(kVideoTag).packets[i];
    cv::Mat output_mat =
        formats::MatView(&(image_frame_packet.Get<ImageFrame>()));
    EXPECT_EQ(1280, output_mat.size().width);
    EXPECT_EQ(640, output_mat.size().height);
    EXPECT_EQ(3, output_mat.channels());
    cv::Scalar s = cv::mean(output_mat);
    for (int i = 0; i < 3; ++i) {
      EXPECT_GT(s[i], 0);
      EXPECT_LT(s[i], 255);
    }
  }
}

TEST(OpenCvVideoDecoderCalculatorTest, TestFlvH264Video) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "OpenCvVideoDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "VIDEO:video"
        output_stream: "VIDEO_PRESTREAM:video_prestream")pb");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag(kInputFilePathTag) =
      MakePacket<std::string>(file::JoinPath(GetTestDataDir(kTestPackageRoot),
                                             "format_FLV_H264_AAC.video"));
  MP_EXPECT_OK(runner.Run());

  EXPECT_EQ(runner.Outputs().Tag(kVideoPrestreamTag).packets.size(), 1);
  MP_EXPECT_OK(runner.Outputs()
                   .Tag(kVideoPrestreamTag)
                   .packets[0]
                   .ValidateAsType<VideoHeader>());
  const mediapipe::VideoHeader& header =
      runner.Outputs().Tag(kVideoPrestreamTag).packets[0].Get<VideoHeader>();
  EXPECT_EQ(ImageFormat::SRGB, header.format);
  EXPECT_EQ(640, header.width);
  EXPECT_EQ(320, header.height);
  // TODO: The actual header.duration is 6.0666666f and the frame_rate
  // can be either 30.30303f (with opencv2) or 30f (with opencv3 and opencv4).
  // EXPECT_FLOAT_EQ(6.0f, header.duration);
  // EXPECT_FLOAT_EQ(30.0f, header.frame_rate);
  EXPECT_EQ(180, runner.Outputs().Tag(kVideoTag).packets.size());
  for (int i = 0; i < 180; ++i) {
    Packet image_frame_packet = runner.Outputs().Tag(kVideoTag).packets[i];
    cv::Mat output_mat =
        formats::MatView(&(image_frame_packet.Get<ImageFrame>()));
    EXPECT_EQ(640, output_mat.size().width);
    EXPECT_EQ(320, output_mat.size().height);
    EXPECT_EQ(3, output_mat.channels());
    cv::Scalar s = cv::mean(output_mat);
    for (int i = 0; i < 3; ++i) {
      EXPECT_GT(s[i], 0);
      EXPECT_LT(s[i], 255);
    }
  }
}

TEST(OpenCvVideoDecoderCalculatorTest, TestMkvVp8Video) {
  CalculatorGraphConfig::Node node_config =
      ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "OpenCvVideoDecoderCalculator"
        input_side_packet: "INPUT_FILE_PATH:input_file_path"
        output_stream: "VIDEO:video"
        output_stream: "VIDEO_PRESTREAM:video_prestream")pb");
  CalculatorRunner runner(node_config);
  runner.MutableSidePackets()->Tag(kInputFilePathTag) =
      MakePacket<std::string>(file::JoinPath(GetTestDataDir(kTestPackageRoot),
                                             "format_MKV_VP8_VORBIS.video"));
  MP_EXPECT_OK(runner.Run());

  EXPECT_EQ(runner.Outputs().Tag(kVideoPrestreamTag).packets.size(), 1);
  MP_EXPECT_OK(runner.Outputs()
                   .Tag(kVideoPrestreamTag)
                   .packets[0]
                   .ValidateAsType<VideoHeader>());
  const mediapipe::VideoHeader& header =
      runner.Outputs().Tag(kVideoPrestreamTag).packets[0].Get<VideoHeader>();
  EXPECT_EQ(ImageFormat::SRGB, header.format);
  EXPECT_EQ(640, header.width);
  EXPECT_EQ(320, header.height);
  EXPECT_FLOAT_EQ(6.0f, header.duration);
  EXPECT_FLOAT_EQ(30.0f, header.frame_rate);
  // The number of the output packets should be 180.
  int num_of_packets = runner.Outputs().Tag(kVideoTag).packets.size();
  EXPECT_GE(num_of_packets, 180);
  for (int i = 0; i < num_of_packets; ++i) {
    Packet image_frame_packet = runner.Outputs().Tag(kVideoTag).packets[i];
    cv::Mat output_mat =
        formats::MatView(&(image_frame_packet.Get<ImageFrame>()));
    EXPECT_EQ(640, output_mat.size().width);
    EXPECT_EQ(320, output_mat.size().height);
    EXPECT_EQ(3, output_mat.channels());
    cv::Scalar s = cv::mean(output_mat);
    for (int i = 0; i < 3; ++i) {
      EXPECT_GT(s[i], 0);
      EXPECT_LT(s[i], 255);
    }
  }
}

}  // namespace
}  // namespace mediapipe
