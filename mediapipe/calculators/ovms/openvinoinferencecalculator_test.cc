//*****************************************************************************
// Copyright 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "openvinoinferencecalculator.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>

#include <adapters/inference_adapter.h>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/calculators/ovms/openvinoinferencecalculator.pb.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/deps/status_matchers.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "tensorflow/core/framework/tensor.h"
#pragma GCC diagnostic pop
using mediapipe::Adopt;
using mediapipe::CalculatorContract;
using mediapipe::CalculatorGraph;
using mediapipe::CalculatorGraphConfig;
using mediapipe::OpenVINOInferenceCalculator;
using mediapipe::CalculatorState;
using mediapipe::ParseTextProtoOrDie;
using mediapipe::Packet;
using mediapipe::PacketType;
using mediapipe::Timestamp;

class OpenVINOInferenceCalculatorTest : public ::testing::Test {
PacketType OVTENSOR_TYPE;
PacketType OVTENSORS_TYPE;
PacketType MPTENSOR_TYPE;
PacketType MPTENSORS_TYPE;
public:
    void SetUp() override {
        OVTENSOR_TYPE.Set<ov::Tensor>();
        OVTENSORS_TYPE.Set<std::vector<ov::Tensor>>();
        MPTENSOR_TYPE.Set<mediapipe::Tensor>();
        MPTENSORS_TYPE.Set<std::vector<mediapipe::Tensor>>();
    }
};

TEST_F(OpenVINOInferenceCalculatorTest, VerifySupportedTags) {
    auto calculator =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:not_used_session"
                input_stream: "OVTENSOR:input"
                input_stream: "MPTENSOR:input"
                input_stream: "TFTENSOR:input"
                input_stream: "TFLITE_TENSOR:input"
                output_stream: "OVTENSORS:output"
                output_stream: "MPTENSORS:output"
                output_stream: "TFTENSORS:output"
                output_stream: "TFLITE_TENSORS:output"
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(calculator);
    auto abslStatus = mediapipe::OpenVINOInferenceCalculator::GetContract(cc.get());
    ASSERT_EQ(abslStatus.code(), absl::StatusCode::kOk) << abslStatus.message();
    EXPECT_EQ(1, cc->InputSidePackets().TagMap()->NumEntries());
    EXPECT_EQ(0, cc->OutputSidePackets().NumEntries());
    auto& inputPacketsTags = cc->Inputs();
    auto& outputPacketsTags = cc->Outputs();
    EXPECT_EQ(4, inputPacketsTags.TagMap()->NumEntries());
    EXPECT_EQ(4, outputPacketsTags.TagMap()->NumEntries());
}
TEST_F(OpenVINOInferenceCalculatorTest, VerifyNotAllowedEmptySideInputPacket) {
    auto calculator =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_stream: "OVTENSOR:input"
                output_stream: "OVTENSOR:output"
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(calculator);
    auto abslStatus = mediapipe::OpenVINOInferenceCalculator::GetContract(cc.get());
    EXPECT_EQ(abslStatus.code(), absl::StatusCode::kInternal) << abslStatus.message();
}
TEST_F(OpenVINOInferenceCalculatorTest, VerifyNotAllowedSideOutputPacket) {
    auto calculator =
        mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                output_side_packet: "SESSION:not_used_session"
                input_stream: "OVTENSOR:input"
                output_stream: "OVTENSOR:output"
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(calculator);
    auto abslStatus = mediapipe::OpenVINOInferenceCalculator::GetContract(cc.get());
    EXPECT_EQ(abslStatus.code(), absl::StatusCode::kInternal) << abslStatus.message();
}
TEST_F(OpenVINOInferenceCalculatorTest, BasicDummyInference) {
    std::string graph_proto = R"(
      input_stream: "input"
      output_stream: "output"
      node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "dummy"
              server_config: "/mediapipe/mediapipe/calculators/ovms/test_data/config.json"
            }
          }
      }
      node {
        calculator: "OpenVINOInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "OVTENSOR:input"
        output_stream: "OVTENSOR:output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                output_order_list: ["Identity:0", "Identity_1:0"]
                tag_to_input_tensor_names {
                    key: "OVTENSOR"
                    value: "b"
                }
                tag_to_output_tensor_names {
                    key: "OVTENSOR"
                    value: "a"
                }
            }
        }
      }
    )";
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
    const std::string inputStreamName = "input";
    const std::string outputStreamName = "output";
    // avoid creating pollers, retreiving packets etc.
    std::vector<Packet> output_packets;
    mediapipe::tool::AddVectorSink(outputStreamName, &graph_config, &output_packets);
    CalculatorGraph graph(graph_config);
    MP_ASSERT_OK(graph.StartRun({}));
    auto datatype = ov::element::Type_t::f32;
    ov::Shape shape{1,10};
    std::vector<float> data{0,1,2,3,4,5,6,7,8,9};
    auto inputTensor = std::make_unique<ov::Tensor>(datatype, shape, data.data());
    MP_ASSERT_OK(graph.AddPacketToInputStream(
        inputStreamName, Adopt(inputTensor.release()).At(Timestamp(0))));
    MP_ASSERT_OK(graph.CloseInputStream(inputStreamName));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    ASSERT_EQ(1, output_packets.size());
    const ov::Tensor& outputTensor =
        output_packets[0].Get<ov::Tensor>();
    MP_ASSERT_OK(graph.WaitUntilDone());
    EXPECT_EQ(datatype, outputTensor.get_element_type());
    EXPECT_THAT(outputTensor.get_shape(), testing::ElementsAre(1,10));
    const void* outputData = outputTensor.data();
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i] + 1, *(reinterpret_cast<const float*>(outputData) + i)) << i;
    }
}
