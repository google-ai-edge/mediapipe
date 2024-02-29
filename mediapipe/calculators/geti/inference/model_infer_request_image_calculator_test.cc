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

#include <fstream>
#include <map>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "../inference/kserve.h"
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
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {

inference::ModelInferRequest build_request(std::string& file_path) {
  auto request = inference::ModelInferRequest();
  std::ifstream is(file_path);
  std::stringstream ss;
  ss << is.rdbuf();

  request.mutable_raw_input_contents()->Add(std::move(ss.str()));
  return request;
}

TEST(ModelInferRequestImageCalculatorTest, ImageIsConvertedToCVMatrix) {
  std::string file_path = "/data/pearl.jpg";

  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input"
            output_stream: "output"
            node {
              calculator: "ModelInferRequestImageCalculator"
              input_stream: "REQUEST:input"
              output_stream: "IMAGE:output"
            }
          )pb"));

  const cv::Mat raw_image = cv::imread(file_path);
  auto request = build_request(file_path);
  auto packet =
      mediapipe::MakePacket<const inference::ModelInferRequest*>(&request);
  std::vector<Packet> output_packets;
  geti::RunGraph(packet, graph_config, output_packets);

  auto& image = output_packets[0].Get<cv::Mat>();
  ASSERT_EQ(image.cols, raw_image.cols);
  ASSERT_EQ(image.rows, image.rows);

  cv::Mat expected_image;
  cv::cvtColor(raw_image, expected_image, cv::COLOR_BGR2RGB);
  bool image_is_identical = !cv::norm(image, expected_image, cv::NORM_L1);
  ASSERT_TRUE(image_is_identical);
}

}  // namespace mediapipe
