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

#include "crop_calculator.h"

#include <vector>

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

bool equal_images(const cv::Mat &a, const cv::Mat &b) {
  if ((a.rows != b.rows) || (a.cols != b.cols)) return false;
  cv::Scalar s = cv::sum(a - b);
  return (s[0] == 0) && (s[1] == 0) && (s[2] == 0);
}

TEST(CropCalculatorTest, TestImageIsCropped) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input_image"
            input_stream: "input_detection_element"
            output_stream: "cropped_image"
            node {
              calculator: "CropCalculator"
              input_stream: "IMAGE:input_image"
              input_stream: "DETECTION:input_detection_element"
              output_stream: "IMAGE:cropped_image"
            }
          )pb"));

  const cv::Mat raw_image = cv::imread("/data/pearl.jpg");
  geti::Label label{"id", "label_name"};
  geti::RectanglePrediction area = {{geti::LabelResult{0.0f, label}},
                                    cv::Rect2f(10, 20, 100, 200)};
  Packet area_packet = MakePacket<geti::RectanglePrediction>(area);
  std::vector<Packet> output_packets;
  tool::AddVectorSink("cropped_image", &graph_config, &output_packets);
  CalculatorGraph graph(graph_config);
  MP_ASSERT_OK(graph.StartRun({}));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input_image",
      mediapipe::MakePacket<cv::Mat>(raw_image).At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.AddPacketToInputStream("input_detection_element",
                                            area_packet.At(Timestamp(0))));
  MP_ASSERT_OK(graph.WaitUntilIdle());
  ASSERT_EQ(1, output_packets.size());

  auto &output_image = output_packets[0].Get<cv::Mat>();
  ASSERT_EQ(output_image.cols, area.shape.width);
  ASSERT_EQ(output_image.rows, area.shape.height);

  ASSERT_TRUE(equal_images(raw_image(area.shape), output_image));
}
}  // namespace mediapipe
