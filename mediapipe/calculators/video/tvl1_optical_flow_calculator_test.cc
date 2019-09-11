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
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/file_path.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/motion/optical_flow_field.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

namespace {
void AddInputPackets(int num_packets, CalculatorGraph* graph) {
  int width = 127;
  int height = 227;
  Packet packet1 = MakePacket<ImageFrame>(ImageFormat::SRGB, width, height);
  Packet packet2 = MakePacket<ImageFrame>(ImageFormat::SRGB, width, height);
  cv::Mat mat1 = formats::MatView(&(packet1.Get<ImageFrame>()));
  cv::Mat mat2 = formats::MatView(&(packet2.Get<ImageFrame>()));
  for (int r = 0; r < mat1.rows; ++r) {
    for (int c = 0; c < mat1.cols; ++c) {
      cv::Vec3b& color1 = mat1.at<cv::Vec3b>(r, c);
      color1[0] = r + 3;
      color1[1] = r + 3;
      color1[2] = 0;
      cv::Vec3b& color2 = mat2.at<cv::Vec3b>(r, c);
      color2[0] = r;
      color2[1] = r;
      color2[2] = 0;
    }
  }

  for (int i = 0; i < num_packets; ++i) {
    MP_ASSERT_OK(graph->AddPacketToInputStream("first_frames",
                                               packet1.At(Timestamp(i))));
    MP_ASSERT_OK(graph->AddPacketToInputStream("second_frames",
                                               packet2.At(Timestamp(i))));
  }
  MP_ASSERT_OK(graph->CloseAllInputStreams());
}

void RunTest(int num_input_packets, int max_in_flight) {
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
      absl::Substitute(R"(
    input_stream: "first_frames"
    input_stream: "second_frames"
    node {
      calculator: "Tvl1OpticalFlowCalculator"
      input_stream: "FIRST_FRAME:first_frames"
      input_stream: "SECOND_FRAME:second_frames"
      output_stream: "FORWARD_FLOW:forward_flow"
      output_stream: "BACKWARD_FLOW:backward_flow"
      max_in_flight: $0
    }
    num_threads: $0
  )",
                       max_in_flight));
  CalculatorGraph graph;
  MP_ASSERT_OK(graph.Initialize(config));
  StatusOrPoller status_or_poller1 =
      graph.AddOutputStreamPoller("forward_flow");
  ASSERT_TRUE(status_or_poller1.ok());
  OutputStreamPoller poller1 = std::move(status_or_poller1.ValueOrDie());
  StatusOrPoller status_or_poller2 =
      graph.AddOutputStreamPoller("backward_flow");
  ASSERT_TRUE(status_or_poller2.ok());
  OutputStreamPoller poller2 = std::move(status_or_poller2.ValueOrDie());

  MP_ASSERT_OK(graph.StartRun({}));
  AddInputPackets(num_input_packets, &graph);
  Packet packet;
  std::vector<Packet> forward_optical_flow_packets;
  while (poller1.Next(&packet)) {
    forward_optical_flow_packets.emplace_back(packet);
  }
  std::vector<Packet> backward_optical_flow_packets;
  while (poller2.Next(&packet)) {
    backward_optical_flow_packets.emplace_back(packet);
  }
  MP_ASSERT_OK(graph.WaitUntilDone());
  EXPECT_EQ(num_input_packets, forward_optical_flow_packets.size());

  int count = 0;
  for (const Packet& packet : forward_optical_flow_packets) {
    cv::Scalar average = cv::mean(packet.Get<OpticalFlowField>().flow_data());
    EXPECT_NEAR(average[0], 0.0, 0.5) << "Actual mean_dx = " << average[0];
    EXPECT_NEAR(average[1], 3.0, 0.5) << "Actual mean_dy = " << average[1];
    EXPECT_EQ(count++, packet.Timestamp().Value());
  }
  EXPECT_EQ(num_input_packets, backward_optical_flow_packets.size());
  count = 0;
  for (const Packet& packet : backward_optical_flow_packets) {
    cv::Scalar average = cv::mean(packet.Get<OpticalFlowField>().flow_data());
    EXPECT_NEAR(average[0], 0.0, 0.5) << "Actual mean_dx = " << average[0];
    EXPECT_NEAR(average[1], -3.0, 0.5) << "Actual mean_dy = " << average[1];
    EXPECT_EQ(count++, packet.Timestamp().Value());
  }
}

TEST(Tvl1OpticalFlowCalculatorTest, TestSequentialExecution) {
  RunTest(/*num_input_packets=*/2, /*max_in_flight=*/1);
}

TEST(Tvl1OpticalFlowCalculatorTest, TestParallelExecution) {
  RunTest(/*num_input_packets=*/20, /*max_in_flight=*/10);
}

}  // namespace
}  // namespace mediapipe
