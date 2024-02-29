/**
 *  INTEL CONFIDENTIAL
 *
 *  Copyright (C) 2023 Intel Corporation
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

#include <map>
#include <string>
#include <vector>

#include "detection_calculator.h"
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
#include "../utils/data_structures.h"

namespace mediapipe {

CalculatorGraphConfig rotated_detection_test_graph =
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
            calculator: "RotatedDetectionCalculator"
            input_side_packet: "INFERENCE_ADAPTER:adapter"
            input_stream: "IMAGE:input"
            output_stream: "INFERENCE_RESULT:output"
          }
        )pb"));

TEST(RotatedDetectionCalculatorTest, TestRotatedDetection) {
  const cv::Mat image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  std::string model_path = "/data/geti/rotated_detection_maskrcnn_resnet50.xml";

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));

  auto packet = mediapipe::MakePacket<cv::Mat>(image);
  geti::RunGraph(packet, rotated_detection_test_graph, output_packets,
                 inputSidePackets);
  const auto &result = output_packets[0].Get<geti::InferenceResult>();
  ASSERT_EQ(result.rotated_rectangles.size(), 9);
  const auto &obj = result.rotated_rectangles[0];
  ASSERT_EQ(obj.labels[0].label.label_id, "653b87ce4e88964031d81d31");

  ASSERT_EQ(result.saliency_maps[0].label.label_id, "653b87ce4e88964031d81d31");

  ASSERT_EQ(result.saliency_maps[1].label.label_id, "653b87ce4e88964031d81d32");
}  // namespace mediapipe

TEST(RotatedDetectionCalculatorTest, TestRotatedDetectionTiler) {
  const cv::Mat image = cv::imread("/data/cattle.jpg");
  std::vector<Packet> output_packets;
  std::string model_path =
      "/data/geti/rotated_detection_maskrcnn_resnet50_tiling.xml";

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));

  auto packet = mediapipe::MakePacket<cv::Mat>(image);
  geti::RunGraph(packet, rotated_detection_test_graph, output_packets,
                 inputSidePackets);
  const auto &result = output_packets[0].Get<geti::InferenceResult>();
  cv::Rect roi(0, 0, image.cols, image.rows);
  ASSERT_EQ(result.roi, roi);
  ASSERT_EQ(result.rotated_rectangles.size(), 18);
  const auto &obj = result.rotated_rectangles[0];
  ASSERT_EQ(obj.labels[0].label.label_id, "65c1ecc04a85ba6e7cc68002");

  ASSERT_EQ(result.saliency_maps[0].label.label_id, "65c1ecc04a85ba6e7cc68002");

  ASSERT_EQ(result.saliency_maps[1].label.label_id, "65c1ecc04a85ba6e7cc68003");
}

}  // namespace mediapipe
