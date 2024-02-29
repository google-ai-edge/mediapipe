//
//  INTEL CONFIDENTIAL
//
//  Copyright (C) 2023 Intel Corporation
//
//  This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
//  This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//

#include "../inference/test_utils.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "../utils/data_structures.h"

namespace mediapipe {

const geti::Label test_label = {"label_id", "test_label_name"};

CalculatorGraphConfig build_graph_config(std::string calculator_name) {
  auto first_part = R"(
        input_stream: "input"
        output_stream: "output"
        node {
          calculator:")";
  auto second_part = R"("
          input_stream: "PREDICTION:input"
          output_stream: "PREDICTION:output"
          node_options: {
            [type.googleapis.com/mediapipe.EmptyLabelOptions] {
              id: "777"
              label: "mytestlabel"
            }
          }
      }
  )";
  std::stringstream ss;
  ss << first_part << calculator_name << second_part << std::endl;
  return ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(ss.str()));
}

TEST(EmptyLabelDetectionCalculatorTest, DetectionOutput) {
  std::vector<Packet> output_packets;

  auto graph_config = build_graph_config("EmptyLabelCalculator");

  geti::InferenceResult inference_result;
  inference_result.rectangles = {
      {{geti::LabelResult{0.0f, test_label}}, cv::Rect2f(10, 10, 10, 10)}};
  geti::RunGraph(MakePacket<geti::InferenceResult>(inference_result),
                 graph_config, output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<geti::InferenceResult>();
  auto& first_object = result.rectangles[0];
  ASSERT_EQ(first_object.labels[0].label.label, test_label.label);
  ASSERT_EQ(first_object.labels[0].label.label_id, test_label.label_id);
  ASSERT_EQ(first_object.labels[0].probability,
            inference_result.rectangles[0].labels[0].probability);
}

TEST(EmptyLabelDetectionCalculatorTest, NoDetectionOutput) {
  std::vector<Packet> output_packets;
  auto graph_config = build_graph_config("EmptyLabelCalculator");

  geti::InferenceResult inference_result;
  inference_result.roi = cv::Rect(0, 0, 256, 128);
  geti::RunGraph(MakePacket<geti::InferenceResult>(inference_result),
                 graph_config, output_packets);

  ASSERT_EQ(1, output_packets.size());

  auto& result = output_packets[0].Get<geti::InferenceResult>();
  auto& first_object = result.rectangles[0];
  ASSERT_EQ(first_object.labels[0].label.label_id, "777");
  ASSERT_EQ(first_object.labels[0].label.label, "mytestlabel");
  ASSERT_EQ(first_object.labels[0].probability, 0);
  ASSERT_EQ(first_object.shape.x, 0);
  ASSERT_EQ(first_object.shape.y, 0);
  ASSERT_EQ(first_object.shape.width, inference_result.roi.width);
  ASSERT_EQ(first_object.shape.height, inference_result.roi.height);
}

}  // namespace mediapipe
