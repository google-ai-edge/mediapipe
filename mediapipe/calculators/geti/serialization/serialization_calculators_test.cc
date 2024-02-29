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

#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json_fwd.hpp>
#include <opencv2/core/base.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "../inference/anomaly_calculator.h"
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
#include "models/results.h"
#include "nlohmann/json.hpp"
#include "third_party/cpp-base64/base64.h"
#include "../utils/data_structures.h"

namespace mediapipe {
inference::ModelInferRequest build_request(std::string file_path,
                                           bool include_xai) {
  auto request = inference::ModelInferRequest();
  std::ifstream is(file_path);
  std::stringstream ss;
  ss << is.rdbuf();

  request.mutable_raw_input_contents()->Add(std::move(ss.str()));
  auto param = inference::InferParameter();
  param.set_bool_param(include_xai);
  (*request.mutable_parameters())["include_xai"] = param;
  return request;
}

static inline void RunGraph(
    mediapipe::Packet request_packet, mediapipe::Packet result_packet,
    std::vector<mediapipe::Packet>& output_packets,
    std::map<std::string, mediapipe::Packet> inputSidePackets = {}) {
  CalculatorGraphConfig graph_config =
      ParseTextProtoOrDie<CalculatorGraphConfig>(absl::Substitute(
          R"pb(
            input_stream: "input"
            input_stream: "result"
            output_stream: "output"
            node {
              calculator: "SerializationCalculator"
              input_stream: "REQUEST:input"
              input_stream: "INFERENCE_RESULT:result"
              output_stream: "RESPONSE:output"
            }
          )pb"));

  mediapipe::tool::AddVectorSink("output", &graph_config, &output_packets);

  mediapipe::CalculatorGraph graph(graph_config);

  MP_ASSERT_OK(graph.StartRun(inputSidePackets));

  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "input", request_packet.At(mediapipe::Timestamp(0))));
  MP_ASSERT_OK(graph.AddPacketToInputStream(
      "result", result_packet.At(mediapipe::Timestamp(0))));

  MP_ASSERT_OK(graph.WaitUntilIdle());
}

TEST(SerializationCalculatorTest, SerializationTestWithoutXAICullsMaps) {
  bool include_xai = false;
  geti::Label test_label{"label_id", "label_name"};
  geti::InferenceResult result;
  cv::Rect roi{0, 0, 100, 100};
  result.rectangles.push_back({{geti::LabelResult{0.0, test_label}}, roi});
  result.saliency_maps.push_back(
      geti::SaliencyMap{cv::Mat::zeros(100, 100, CV_32FC1), roi, test_label});

  mediapipe::Packet result_packet = MakePacket<geti::InferenceResult>(result);
  auto request = build_request("/data/cattle.jpg", include_xai);
  mediapipe::Packet request_packet =
      mediapipe::MakePacket<const KFSRequest*>(&request);

  std::vector<Packet> output_packets;

  RunGraph(request_packet, result_packet, output_packets);

  ASSERT_EQ(1, output_packets.size());
  auto& response = output_packets[0].Get<KFSResponse*>();
  auto& actual_string = response->parameters().at("predictions").string_param();
  nlohmann::json actual = nlohmann::json::parse(actual_string);

  ASSERT_EQ(1, actual["predictions"].size());
  ASSERT_EQ(test_label.label_id, actual["predictions"][0]["labels"][0]["id"]);
  ASSERT_EQ(0, actual.count("maps"));
}

TEST(SerializationCalculatorTest, SerializationTestWithXAIReturnsMaps) {
  bool include_xai = true;
  geti::Label test_label{"label_id", "label_name"};
  geti::InferenceResult result;
  cv::Rect roi{0, 0, 100, 100};
  result.rectangles.push_back({{geti::LabelResult{0.0, test_label}}, roi});
  result.saliency_maps.push_back(
      geti::SaliencyMap{cv::Mat::zeros(100, 100, CV_32FC1), roi, test_label});

  mediapipe::Packet result_packet = MakePacket<geti::InferenceResult>(result);
  auto request = build_request("/data/cattle.jpg", include_xai);
  mediapipe::Packet request_packet =
      mediapipe::MakePacket<const KFSRequest*>(&request);

  std::vector<Packet> output_packets;

  RunGraph(request_packet, result_packet, output_packets);

  ASSERT_EQ(1, output_packets.size());
  auto& response = output_packets[0].Get<KFSResponse*>();
  auto& actual_string = response->parameters().at("predictions").string_param();
  nlohmann::json actual = nlohmann::json::parse(actual_string);

  ASSERT_EQ(1, actual["maps"].size());
  ASSERT_EQ(test_label.label_id, actual["maps"][0]["label_id"]);
}

TEST(SerializationCalculatorTest,
     SerializationTestWithXAICullsMapsWithoutPrediction) {
  bool include_xai = true;
  geti::Label test_label{"label_id", "label_name"};
  geti::InferenceResult result;
  cv::Rect roi{0, 0, 100, 100};
  result.saliency_maps.push_back(
      geti::SaliencyMap{cv::Mat::zeros(100, 100, CV_32FC1), roi, test_label});

  mediapipe::Packet result_packet = MakePacket<geti::InferenceResult>(result);
  auto request = build_request("/data/cattle.jpg", include_xai);
  mediapipe::Packet request_packet =
      mediapipe::MakePacket<const KFSRequest*>(&request);

  std::vector<Packet> output_packets;

  RunGraph(request_packet, result_packet, output_packets);

  ASSERT_EQ(1, output_packets.size());
  auto& response = output_packets[0].Get<KFSResponse*>();
  auto& actual_string = response->parameters().at("predictions").string_param();
  nlohmann::json actual = nlohmann::json::parse(actual_string);

  ASSERT_EQ(0, actual["maps"].size());
}

}  // namespace mediapipe
