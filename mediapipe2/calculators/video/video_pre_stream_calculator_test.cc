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
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace {

TEST(VideoPreStreamCalculatorTest, ProcessesWithFrameRateInOptions) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "input"
    node {
      calculator: "VideoPreStreamCalculator"
      input_stream: "input"
      output_stream: "output"
      options {
        [mediapipe.VideoPreStreamCalculatorOptions.ext] { fps { value: 3 } }
      }
    })pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  auto poller_status = graph.AddOutputStreamPoller("output");
  MP_ASSERT_OK(poller_status.status());
  OutputStreamPoller& poller = poller_status.value();
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input",
      Adopt(new ImageFrame(ImageFormat::SRGB, 1, 2)).At(Timestamp(0))));

  // It is *not* VideoPreStreamCalculator's job to detect errors in an
  // ImageFrame stream.  It just waits for the 1st ImageFrame, extracts info for
  // VideoHeader, and emits it.  Thus, the following is fine.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input",
      Adopt(new ImageFrame(ImageFormat::SRGBA, 3, 4)).At(Timestamp(1))));

  MP_ASSERT_OK(graph.CloseInputStream("input"));
  Packet packet;
  ASSERT_TRUE(poller.Next(&packet));
  const auto& video_header = packet.Get<VideoHeader>();
  EXPECT_EQ(video_header.format, ImageFormat::SRGB);
  EXPECT_EQ(video_header.width, 1);
  EXPECT_EQ(video_header.height, 2);
  EXPECT_EQ(video_header.frame_rate, 3);
  EXPECT_EQ(packet.Timestamp(), Timestamp::PreStream());
  ASSERT_FALSE(poller.Next(&packet));
  MP_EXPECT_OK(graph.WaitUntilDone());
}

TEST(VideoPreStreamCalculatorTest, ProcessesWithFrameRateInPreStream) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "frame"
    input_stream: "input_header"
    node {
      calculator: "VideoPreStreamCalculator"
      input_stream: "FRAME:frame"
      input_stream: "VIDEO_PRESTREAM:input_header"
      output_stream: "output_header"
    })pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  auto poller_status = graph.AddOutputStreamPoller("output_header");
  MP_ASSERT_OK(poller_status.status());
  OutputStreamPoller& poller = poller_status.value();
  MP_ASSERT_OK(graph.StartRun({}));
  auto input_header = absl::make_unique<VideoHeader>();
  input_header->frame_rate = 3.0;
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_header",
      Adopt(input_header.release()).At(Timestamp::PreStream())));
  MP_ASSERT_OK(graph.CloseInputStream("input_header"));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "frame",
      Adopt(new ImageFrame(ImageFormat::SRGB, 1, 2)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("frame"));
  Packet packet;
  ASSERT_TRUE(poller.Next(&packet));
  const auto& output_header = packet.Get<VideoHeader>();
  EXPECT_EQ(output_header.format, ImageFormat::SRGB);
  EXPECT_EQ(output_header.width, 1);
  EXPECT_EQ(output_header.height, 2);
  EXPECT_EQ(output_header.frame_rate, 3.0);
  EXPECT_EQ(packet.Timestamp(), Timestamp::PreStream());
  ASSERT_FALSE(poller.Next(&packet));
  MP_EXPECT_OK(graph.WaitUntilDone());
}

TEST(VideoPreStreamCalculatorTest, FailsWithoutFrameRateInOptions) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "frame"
    node {
      calculator: "VideoPreStreamCalculator"
      input_stream: "frame"
      output_stream: "output_header"
    })pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "frame",
      Adopt(new ImageFrame(ImageFormat::SRGB, 1, 2)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("frame"));
  absl::Status status = graph.WaitUntilDone();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("frame rate should be non-zero"));
}

// Input header missing.
TEST(VideoPreStreamCalculatorTest, FailsWithoutFrameRateInPreStream1) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "frame"
    input_stream: "input_header"
    node {
      calculator: "VideoPreStreamCalculator"
      input_stream: "FRAME:frame"
      input_stream: "VIDEO_PRESTREAM:input_header"
      output_stream: "output_header"
    }
  )pb");
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "frame",
      Adopt(new ImageFrame(ImageFormat::SRGB, 1, 2)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.CloseInputStream("frame"));
  MP_ASSERT_OK(graph.CloseInputStream("input_header"));
  absl::Status status = graph.WaitUntilDone();
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(),
              testing::HasSubstr("frame rate should be non-zero"));
}

// Input header not at prestream (before, with, and after frame data).
TEST(VideoPreStreamCalculatorTest, FailsWithoutFrameRateInPreStream2) {
  auto config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"pb(
    input_stream: "frame"
    input_stream: "input_header"
    node {
      calculator: "VideoPreStreamCalculator"
      input_stream: "FRAME:frame"
      input_stream: "VIDEO_PRESTREAM:input_header"
      output_stream: "output_header"
    }
  )pb");

  for (int64 timestamp = -1; timestamp < 2; ++timestamp) {
    CalculatorGraph graph;
    MP_ASSERT_OK(graph.Initialize(config));
    MP_ASSERT_OK(graph.StartRun({}));
    auto input_header = absl::make_unique<VideoHeader>();
    input_header->frame_rate = 3.0;
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "input_header",
        Adopt(input_header.release()).At(Timestamp(timestamp))));
    MP_ASSERT_OK(graph.CloseInputStream("input_header"));
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        "frame",
        Adopt(new ImageFrame(ImageFormat::SRGB, 1, 2)).At(Timestamp(0))));
    MP_ASSERT_OK(graph.CloseInputStream("frame"));
    absl::Status status = graph.WaitUntilDone();
    EXPECT_FALSE(status.ok());
  }
}

}  // namespace
}  // namespace mediapipe
