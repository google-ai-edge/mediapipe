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

#include "instance_segmentation_calculator.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>
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
#include "models/image_model.h"
#include "../utils/data_structures.h"

namespace mediapipe {

TEST(InstanceSegmentationCalculatorTest, TestImageSegmentation) {
  const cv::Mat image = cv::imread("/data/cattle.jpg");
  const std::string model_path =
      "/data/geti/instance_segmentation_maskrcnn_resnet50.xml";

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
              calculator: "InstanceSegmentationCalculator"
              input_side_packet: "INFERENCE_ADAPTER:adapter"
              input_stream: "IMAGE:input"
              output_stream: "INFERENCE_RESULT:output"
            }
          )pb"));

  std::vector<Packet> output_packets;
  auto packet = mediapipe::MakePacket<cv::Mat>(image);

  std::map<std::string, mediapipe::Packet> inputSidePackets;
  inputSidePackets["model_path"] =
      mediapipe::MakePacket<std::string>(model_path)
          .At(mediapipe::Timestamp(0));
  inputSidePackets["device"] =
      mediapipe::MakePacket<std::string>("AUTO").At(mediapipe::Timestamp(0));

  geti::RunGraph(packet, graph_config, output_packets, inputSidePackets);

  auto &result = output_packets[0].Get<geti::InferenceResult>();
  ASSERT_EQ(18, result.polygons.size());

  cv::Rect roi(0, 0, image.cols, image.rows);
  ASSERT_EQ(result.roi, roi);
  ASSERT_EQ(result.polygons[0].labels[0].label.label_id,
            "653b85cb4e88964031d81b35");
}

}  // namespace mediapipe
