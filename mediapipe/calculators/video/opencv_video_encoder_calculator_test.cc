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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/deleting_file.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {
// Temporarily disable the test.
// TODO: Investigate the “Could not open codec 'libx264'” error with
// opencv2.
TEST(OpenCvVideoEncoderCalculatorTest, DISABLED_TestMp4Avc720pVideo) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    node {
      calculator: "OpenCvVideoDecoderCalculator"
      input_side_packet: "INPUT_FILE_PATH:input_file_path"
      output_stream: "VIDEO:video"
      output_stream: "VIDEO_PRESTREAM:video_prestream"
    }
    node {
      calculator: "OpenCvVideoEncoderCalculator"
      input_stream: "VIDEO:video"
      input_stream: "VIDEO_PRESTREAM:video_prestream"
      input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
      node_options {
        [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
          codec: "avc1"
          video_format: "mp4"
        }
      }
    }
  )");
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["input_file_path"] = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/video/"
                     "testdata/format_MP4_AVC720P_AAC.video"));
  const std::string output_file_path = "/tmp/tmp_video.mp4";
  DeletingFile deleting_file(output_file_path, true);
  input_side_packets["output_file_path"] =
      MakePacket<std::string>(output_file_path);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config, input_side_packets));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("video_prestream");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));
  Packet packet;
  while (poller.Next(&packet)) {
  }
  MP_ASSERT_OK(graph.WaitUntilDone());
  const VideoHeader& video_header = packet.Get<VideoHeader>();

  // Checks the generated video file has the same width, height, fps, and
  // duration as the original one.
  cv::VideoCapture cap(output_file_path);
  ASSERT_TRUE(cap.isOpened());
  EXPECT_EQ(video_header.width,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
  EXPECT_EQ(video_header.height,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  EXPECT_EQ(video_header.frame_rate,
            static_cast<double>(cap.get(cv::CAP_PROP_FPS)));
  EXPECT_EQ(video_header.duration,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT) /
                             cap.get(cv::CAP_PROP_FPS)));
}

TEST(OpenCvVideoEncoderCalculatorTest, TestFlvH264Video) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    node {
      calculator: "OpenCvVideoDecoderCalculator"
      input_side_packet: "INPUT_FILE_PATH:input_file_path"
      output_stream: "VIDEO:video"
      output_stream: "VIDEO_PRESTREAM:video_prestream"
    }
    node {
      calculator: "OpenCvVideoEncoderCalculator"
      input_stream: "VIDEO:video"
      input_stream: "VIDEO_PRESTREAM:video_prestream"
      input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
      node_options {
        [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
          codec: "MJPG"
          video_format: "avi"
        }
      }
    }
  )");
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["input_file_path"] = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/video/"
                     "testdata/format_FLV_H264_AAC.video"));
  const std::string output_file_path = "/tmp/tmp_video.avi";
  DeletingFile deleting_file(output_file_path, true);
  input_side_packets["output_file_path"] =
      MakePacket<std::string>(output_file_path);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config, input_side_packets));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("video_prestream");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));
  Packet packet;
  while (poller.Next(&packet)) {
  }
  MP_ASSERT_OK(graph.WaitUntilDone());
  const VideoHeader& video_header = packet.Get<VideoHeader>();

  // Checks the generated video file has the same width, height, fps, and
  // duration as the original one.
  cv::VideoCapture cap(output_file_path);
  ASSERT_TRUE(cap.isOpened());
  EXPECT_EQ(video_header.width,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
  EXPECT_EQ(video_header.height,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  // TODO: The actual header.duration is 6.0666666f and the frame_rate
  // can be either 30.30303f (with opencv2) or 30f (with opencv3 and opencv4).
  // EXPECT_EQ(video_header.frame_rate,
  //           static_cast<double>(cap.get(cv::CAP_PROP_FPS)));
  // EXPECT_EQ(video_header.duration,
  //           static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT) /
  //                            cap.get(cv::CAP_PROP_FPS)));
}

TEST(OpenCvVideoEncoderCalculatorTest, TestMkvVp8Video) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
    node {
      calculator: "OpenCvVideoDecoderCalculator"
      input_side_packet: "INPUT_FILE_PATH:input_file_path"
      output_stream: "VIDEO:video"
      output_stream: "VIDEO_PRESTREAM:video_prestream"
    }
    node {
      calculator: "OpenCvVideoEncoderCalculator"
      input_stream: "VIDEO:video"
      input_stream: "VIDEO_PRESTREAM:video_prestream"
      input_side_packet: "OUTPUT_FILE_PATH:output_file_path"
      node_options {
        [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
          codec: "PIM1"
          video_format: "mkv"
        }
      }
    }
  )");
  std::map<std::string, Packet> input_side_packets;
  input_side_packets["input_file_path"] = MakePacket<std::string>(
      file::JoinPath("./",
                     "/mediapipe/calculators/video/"
                     "testdata/format_MKV_VP8_VORBIS.video"));
  const std::string output_file_path = "/tmp/tmp_video.mkv";
  DeletingFile deleting_file(output_file_path, true);
  input_side_packets["output_file_path"] =
      MakePacket<std::string>(output_file_path);
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config, input_side_packets));
  StatusOrPoller status_or_poller =
      graph.AddOutputStreamPoller("video_prestream");
  ASSERT_TRUE(status_or_poller.ok());
  OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));
  Packet packet;
  while (poller.Next(&packet)) {
  }
  MP_ASSERT_OK(graph.WaitUntilDone());
  const VideoHeader& video_header = packet.Get<VideoHeader>();

  // Checks the generated video file has the same width, height, fps, and
  // duration as the original one.
  cv::VideoCapture cap(output_file_path);
  ASSERT_TRUE(cap.isOpened());
  EXPECT_EQ(video_header.width,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
  EXPECT_EQ(video_header.height,
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
  EXPECT_EQ(video_header.frame_rate,
            static_cast<double>(cap.get(cv::CAP_PROP_FPS)));
  EXPECT_EQ(video_header.duration,
            static_cast<int>(std::round(cap.get(cv::CAP_PROP_FRAME_COUNT) /
                                        cap.get(cv::CAP_PROP_FPS))));
}

}  // namespace
}  // namespace mediapipe
