/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2024 Intel Corporation
 *
 *  This software and the related documents are Intel copyrighted materials, and
 * your use of them is governed by the express license under which they were
 * provided to you ("License"). Unless the License provides otherwise, you may
 * not use, modify, copy, publish, distribute, disclose or transmit this
 * software or the related documents without Intel's prior written permission.
 *
 *  This software and the related documents are provided as is, with no express
 * or implied warranties, other than those that are expressly stated in the
 * License.
 */

#include "keypoint_detection_calculator.h"

#include <map>
#include <string>
#include <vector>

#include "../inference/test_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {

TEST(KeypointDetectionCalculatorTest, TestDetection) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input"
            input_side_packet: "model_path"
            input_side_packet: "device"
            output_stream: "output"
            node {
              calculator: "OpenVINOInferenceAdapterCalculator"
              input_side_packet: "MODEL_PATH:model_path"
              input_side_packet: "DEVICE:device"
              output_side_packet: "INFERENCE_ADAPTER:adapter"
            }
            node {
              calculator: "KeypointDetectionCalculator"
              input_side_packet: "INFERENCE_ADAPTER:adapter"
              input_stream: "IMAGE:input"
              output_stream: "INFERENCE_RESULT:output"
            }
          )pb"));

  const cv::Mat raw_image = cv::imread("/data/tennis.jpg");
  cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);
  std::vector<Packet> output_packets;
  std::string model_path =
      "/data/omz_models/public/rtmpose_tiny/rtmpose_tiny.xml";

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));
  geti::RunGraph(mediapipe::MakePacket<cv::Mat>(raw_image), graph_config,
                 output_packets, inputSidePackets);
  ASSERT_EQ(1, output_packets.size());

  auto result = output_packets[0].Get<geti::InferenceResult>();
  std::vector<geti::DetectedKeypoints> poses = result.poses;

  ASSERT_EQ(poses.size(), 17);

  cv::Point3f expected[17] = {
      {246.7f, 101.8f, 0.985f}, {238.3f, 83.6f, 1.058f},
      {238.3f, 83.6f, 1.067f},  {238.3f, 82.8f, 0.834f},
      {221.7f, 82.8f, 1.156f},  {225.0f, 105.6f, 0.697f},
      {201.7f, 114.7f, 0.956f}, {246.7f, 151.2f, 0.819f},
      {198.3f, 152.7f, 1.075f}, {280.0f, 162.6f, 0.774f},
      {246.7f, 172.5f, 0.645f}, {246.7f, 180.8f, 0.778f},
      {200.0f, 180.1f, 0.629f}, {248.3f, 224.1f, 0.722f},
      {236.7f, 240.1f, 0.906f}, {193.3f, 286.4f, 0.683f},
      {185.0f, 298.6f, 0.810f}};

  for (int i = 0; i < 17; i++) {
    ASSERT_NEAR(poses[i].shape.x, expected[i].x, 0.1f);
    ASSERT_NEAR(poses[i].shape.y, expected[i].y, 0.1f);
    ASSERT_NEAR(poses[i].labels[0].probability, expected[i].z, 0.001f);
  }
}

}  // namespace mediapipe
