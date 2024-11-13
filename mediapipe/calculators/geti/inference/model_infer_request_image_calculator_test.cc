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
#include <utility>
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

const auto graph_content = R"pb(
  input_stream: "input"
  output_stream: "output"
  node {
    calculator: "ModelInferRequestImageCalculator"
    input_stream: "REQUEST:input"
    output_stream: "IMAGE:output"
  }
)pb";

inference::ModelInferRequest build_request(const std::string& file_path) {
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
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(graph_content));

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

TEST(ModelInferRequestImageCalculatorTest, WebpIsConvertedToCVMatrix) {
  std::string file_path = "/data/pearl.webp";

  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(graph_content));

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

TEST(ModelInferRequestImageCalculatorTest, ImageTooSmallThrowsError) {
  testing::internal::CaptureStdout();
  std::string file_path = "/data/pearl.jpg";

  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(graph_content));

  const cv::Mat raw_image = cv::imread(file_path);
  cv::Mat too_small_image;
  cv::resize(raw_image, too_small_image, cv::Size(25, 25));

  std::vector<uchar> buffer;
  cv::imencode(".jpg", too_small_image, buffer);

  std::string image_data(buffer.begin(), buffer.end());

  auto request = inference::ModelInferRequest();
  request.mutable_raw_input_contents()->Add(std::move(image_data));
  auto packet =
      mediapipe::MakePacket<const inference::ModelInferRequest*>(&request);
  std::vector<Packet> output_packets;
  mediapipe::tool::AddVectorSink("output", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);

  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", packet.At(mediapipe::Timestamp(0))));

  auto status = graph.WaitUntilIdle();
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(0, output_packets.size());
  std::string output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output,
            "Caught exception with message: IMAGE_SIZE_OUT_OF_BOUNDS\n");
}

TEST(ModelInferRequestImageCalculatorTest, ImageTooBigThrowsError) {
  testing::internal::CaptureStdout();
  std::string file_path = "/data/pearl.jpg";

  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(
          absl::Substitute(graph_content));

  const cv::Mat raw_image = cv::imread(file_path);
  cv::Mat too_big_image;
  cv::resize(raw_image, too_big_image, cv::Size(8000, 8000));

  std::vector<uchar> buffer;
  cv::imencode(".jpg", too_big_image, buffer);

  std::string image_data(buffer.begin(), buffer.end());

  auto request = inference::ModelInferRequest();
  request.mutable_raw_input_contents()->Add(std::move(image_data));
  auto packet =
      mediapipe::MakePacket<const inference::ModelInferRequest*>(&request);
  std::vector<Packet> output_packets;
  mediapipe::tool::AddVectorSink("output", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);

  MP_ASSERT_OK(graph.StartRun({}));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", packet.At(mediapipe::Timestamp(0))));

  auto status = graph.WaitUntilIdle();
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(0, output_packets.size());
  std::string output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output,
            "Caught exception with message: IMAGE_SIZE_OUT_OF_BOUNDS\n");
}

}  // namespace mediapipe
