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

#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {

TEST(TimestampAlignInputStreamHandlerTest, Initialization) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_video"
        input_stream: "input_camera"
        node {
          calculator: "PassThroughCalculator"
          input_stream: "VIDEO:input_video"
          input_stream: "CAMERA:input_camera"
          output_stream: "VIDEO:output_video"
          output_stream: "CAMERA:output_camera"
          input_stream_handler {
            input_stream_handler: "TimestampAlignInputStreamHandler"
            options: {
              [mediapipe.TimestampAlignInputStreamHandlerOptions.ext]: {
                timestamp_base_tag_index: "CAMERA"
              }
            }
          }
        })");
  std::vector<Packet> sink_video, sink_camera;
  tool::AddVectorSink("output_video", &config, &sink_video);
  tool::AddVectorSink("output_camera", &config, &sink_camera);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(1)).At(Timestamp(101))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // The timestamp base stream's packet is output immediately.
  EXPECT_EQ(0, sink_video.size());
  ASSERT_EQ(1, sink_camera.size());
  EXPECT_EQ(1, sink_camera[0].Get<int>());
  EXPECT_EQ(Timestamp(101), sink_camera[0].Timestamp());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(2)).At(Timestamp(102))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // The timestamp base stream's packet is output immediately.
  EXPECT_EQ(0, sink_video.size());
  ASSERT_EQ(2, sink_camera.size());
  EXPECT_EQ(2, sink_camera[1].Get<int>());
  EXPECT_EQ(Timestamp(102), sink_camera[1].Timestamp());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_video", Adopt(new int(1)).At(Timestamp(1))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // No packet is output. The packet added to input_video is buffered in the
  // input stream.
  EXPECT_EQ(0, sink_video.size());
  EXPECT_EQ(2, sink_camera.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(3)).At(Timestamp(103))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Both input streams have a packet. The following equivalence of timestamps
  // is established:
  //   input_video    input_camera
  //   1              103
  //
  // The input stream handler is now initialized. From now on, it operates
  // like the default input stream handler except that timestamp offsets are
  // applied.
  ASSERT_EQ(1, sink_video.size());
  ASSERT_EQ(3, sink_camera.size());
  EXPECT_EQ(1, sink_video[0].Get<int>());
  EXPECT_EQ(Timestamp(103), sink_video[0].Timestamp());
  EXPECT_EQ(3, sink_camera[2].Get<int>());
  EXPECT_EQ(Timestamp(103), sink_camera[2].Timestamp());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(4)).At(Timestamp(104))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // The timestamp base stream does not receive special treatment now.
  EXPECT_EQ(1, sink_video.size());
  EXPECT_EQ(3, sink_camera.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_video", Adopt(new int(4)).At(Timestamp(4))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(1, sink_video.size());
  ASSERT_EQ(4, sink_camera.size());
  EXPECT_EQ(4, sink_camera[3].Get<int>());
  EXPECT_EQ(Timestamp(104), sink_camera[3].Timestamp());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(2, sink_video.size());
  EXPECT_EQ(4, sink_camera.size());
  EXPECT_EQ(4, sink_video[1].Get<int>());
  EXPECT_EQ(Timestamp(106), sink_video[1].Timestamp());
}

TEST(TimestampAlignInputStreamHandlerTest, TickRate) {
  CalculatorGraphConfig config =
      ::mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "input_video"
        input_stream: "input_camera"
        node {
          calculator: "PacketClonerCalculator"
          input_stream: "input_camera"
          input_stream: "input_video"
          input_stream: "input_video"
          output_stream: "output_camera"
          output_stream: "output_video"
          input_stream_handler {
            input_stream_handler: "TimestampAlignInputStreamHandler"
            options: {
              [mediapipe.TimestampAlignInputStreamHandlerOptions.ext]: {
                timestamp_base_tag_index: ":0"  # input_camera
              }
            }
          }
        })");
  std::vector<Packet> sink_video, sink_camera;
  tool::AddVectorSink("output_video", &config, &sink_video);
  tool::AddVectorSink("output_camera", &config, &sink_camera);

  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  MP_ASSERT_OK(graph.StartRun({}));

  // Video timestamps start from 0 seconds. Video frame rate is 2 fps.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_video", Adopt(new int(0)).At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // No packets expected as the timestamp base stream has not seen any packet.
  EXPECT_EQ(0, sink_video.size());
  EXPECT_EQ(0, sink_camera.size());

  // Camera timestamps start from 100 seconds. Camera frame rate is 1 fps.
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(0)).At(Timestamp(100000000))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  // Both input streams have a packet. The following equivalence of timestamps
  // is established:
  //   input_video    input_camera
  //   0              100000000
  ASSERT_EQ(1, sink_video.size());
  ASSERT_EQ(1, sink_camera.size());
  EXPECT_EQ(0, sink_video[0].Get<int>());
  EXPECT_EQ(Timestamp(100000000), sink_video[0].Timestamp());
  EXPECT_EQ(0, sink_camera[0].Get<int>());
  EXPECT_EQ(Timestamp(100000000), sink_camera[0].Timestamp());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_video", Adopt(new int(1)).At(Timestamp(500000))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  EXPECT_EQ(1, sink_video.size());
  EXPECT_EQ(1, sink_camera.size());

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_video", Adopt(new int(2)).At(Timestamp(1000000))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_camera", Adopt(new int(1)).At(Timestamp(101000000))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(3, sink_video.size());
  ASSERT_EQ(3, sink_camera.size());
  EXPECT_EQ(1, sink_video[1].Get<int>());
  EXPECT_EQ(Timestamp(100500000), sink_video[1].Timestamp());
  EXPECT_EQ(0, sink_camera[1].Get<int>());
  EXPECT_EQ(Timestamp(100500000), sink_camera[1].Timestamp());
  EXPECT_EQ(2, sink_video[2].Get<int>());
  EXPECT_EQ(Timestamp(101000000), sink_video[2].Timestamp());
  EXPECT_EQ(1, sink_camera[2].Get<int>());
  EXPECT_EQ(Timestamp(101000000), sink_camera[2].Timestamp());

  MP_ASSERT_OK(graph.CloseAllInputStreams());
  MP_ASSERT_OK(graph.WaitUntilDone());
  ASSERT_EQ(3, sink_video.size());
  ASSERT_EQ(3, sink_camera.size());
}

}  // namespace
}  // namespace mediapipe
