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
#include <chrono>
#include <thread>
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
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions","3","4"]
                    }
                }
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

void runDummyInference(std::string& graph_proto) {
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
    runDummyInference(graph_proto);
}
TEST_F(OpenVINOInferenceCalculatorTest, BasicDummyInferenceEmptyKey) {
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
        input_stream: "input"
        output_stream: "output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                tag_to_input_tensor_names {
                    key: ""
                    value: "b"
                }
                tag_to_output_tensor_names {
                    key: ""
                    value: "a"
                }
            }
        }
      }
    )";
    runDummyInference(graph_proto);
}
TEST_F(OpenVINOInferenceCalculatorTest, HandleEmptyPackets) {
    std::string graph_proto = R"(
      input_stream: "input"
      input_stream: "input2"
      output_stream: "output"
      node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "add_two_inputs"
              server_config: "/mediapipe/mediapipe/calculators/ovms/test_data/config.json"
            }
          }
      }
      node {
        calculator: "OpenVINOInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "OVTENSOR:input"
        input_stream: "OVTENSOR2:input2" # we don't expect that in a model but calculator will try to deserialize that
        output_stream: "OVTENSOR:output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                tag_to_input_tensor_names {
                    key: "OVTENSOR"
                    value: "input1"
                }
                tag_to_input_tensor_names {
                    key: "OVTENSOR2"
                    value: "input2"
                }
                tag_to_output_tensor_names {
                    key: "OVTENSOR"
                    value: "sum"
                }
            }
        }
      }
    )";
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
    const std::string inputStreamName = "input";
    const std::string input2StreamName = "input2";
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
    MP_ASSERT_OK(graph.WaitUntilIdle());
    MP_ASSERT_OK(graph.CloseInputStream(inputStreamName));
    MP_ASSERT_OK(graph.CloseInputStream(input2StreamName));
    EXPECT_EQ(graph.WaitUntilDone().code(), absl::StatusCode::kInternal);
    ASSERT_EQ(0, output_packets.size());
    ASSERT_EQ(0, output_packets.size());
}

TEST_F(OpenVINOInferenceCalculatorTest, DISABLED_HandleEmptyPacketsWithSyncSet) {
    std::string graph_proto = R"(
      input_stream: "input"
      input_stream: "input2"
      output_stream: "output"
      node {
          calculator: "OpenVINOModelServerSessionCalculator"
          output_side_packet: "SESSION:session"
          node_options: {
            [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
              servable_name: "add_two_inputs"
              server_config: "/mediapipe/mediapipe/calculators/ovms/test_data/config.json"
            }
          }
      }
      node {
        calculator: "OpenVINOInferenceCalculator"
        input_side_packet: "SESSION:session"
        input_stream: "OVTENSOR:input"
        input_stream: "OVTENSOR2:input2" # we don't expect that in a model but calculator will try to deserialize that
        input_stream_handler {
            input_stream_handler: "SyncSetInputStreamHandler",
            options {
                [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
                    sync_set {
                        tag_index: "OVTENSOR"
                        tag_index: "OVTENSOR2"
                    }
                }
            }
        }
        output_stream: "OVTENSOR:output"
        node_options: {
            [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                tag_to_input_tensor_names {
                    key: "OVTENSOR"
                    value: "input1"
                }
                tag_to_input_tensor_names {
                    key: "OVTENSOR2"
                    value: "input2"
                }
                tag_to_output_tensor_names {
                    key: "OVTENSOR"
                    value: "sum"
                }
            }
        }
      }
    )";
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(graph_proto);
    const std::string inputStreamName = "input";
    const std::string input2StreamName = "input2";
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
    auto inputTensor2 = std::make_unique<ov::Tensor>(datatype, shape, data.data());

    MP_ASSERT_OK(graph.AddPacketToInputStream(
        inputStreamName, Adopt(inputTensor.release()).At(Timestamp(0))));
    MP_ASSERT_OK(graph.WaitUntilIdle());
    MP_ASSERT_OK(graph.CloseInputStream(inputStreamName));
    MP_ASSERT_OK(graph.CloseInputStream(input2StreamName));
    EXPECT_EQ(graph.WaitUntilDone().code(), absl::StatusCode::kOk);
    ASSERT_EQ(0, output_packets.size());
    ASSERT_EQ(0, output_packets.size());
}

void verifyGetContract(const std::string& pbtxtContent, absl::StatusCode expectedStatusCode) {
    auto calculator = mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(pbtxtContent);
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(calculator);
    auto abslStatus = mediapipe::OpenVINOInferenceCalculator::GetContract(cc.get());
    EXPECT_EQ(abslStatus.code(), expectedStatusCode) << abslStatus.message();
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyTagToInputNames) {
    // Test passes with OVTENSORS1 in tag_to_output_tensor_names because we support and check the basic type match - OVTENSORS in this case
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSOR:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "OVTENSOR"
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";

    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsInputFail) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSORS:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        tag_to_input_tensor_names {
                            key: "OVTENSOR"
                            value: "normalized_input_image_tensor"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsOutputFail) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSORS:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        tag_to_output_tensor_names {
                            key: "OVTENSOR1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsInputFailSingleType) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSORS:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsOutputFailSingleType) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSORS:image_tensor"
                output_stream: "OVTENSOR:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsInput) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSORS:image_tensor"
                output_stream: "OVTENSOR2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        tag_to_output_tensor_names {
                            key: "OVTENSOR1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, VerifyOptionsOutput) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSORS2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        tag_to_input_tensor_names {
                            key: "OVTENSOR"
                            value: "normalized_input_image_tensor"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToOutputNames) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSOR2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "OVTENSOR"
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "RROVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToInputNames) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSOR2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "OVTENSORS"
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToInputNamesNoVector) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSOR:image_tensor"
                output_stream: "OVTENSOR2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "OVTENSORS"
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "OVTENSOR2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToInputNamesNoTypeSpecifiedWithMatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "image_tensor"
                output_stream: "detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: ""
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: ""
                            value: "raw_outputs/box_encodings"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToOutputNamesNoTypeSpecifiedWithoutMatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "image_tensor"
                output_stream: "detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "image_tensor"
                            value: "normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "BAD_detection_tensors1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "detection_tensors2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, WrongTagToInputNamesNoTypeSpecifiedWithoutMatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "image_tensor"
                output_stream: "detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        tag_to_input_tensor_names {
                            key: "image_tensor"
                            value: "BAD_normalized_input_image_tensor"
                        }
                        tag_to_output_tensor_names {
                            key: "detection_tensors1"
                            value: "raw_outputs/box_encodings"
                        }
                        tag_to_output_tensor_names {
                            key: "detection_tensors2"
                            value: "raw_outputs/class_predictions"
                        }
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, NoTagToInputNames) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "OVTENSORS:image_tensor"
                output_stream: "OVTENSORS2:detection_tensors"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                        input_order_list :["normalized_input_image_tensor"]
                        output_order_list :["raw_outputs/box_encodings","raw_outputs/class_predictions"]
                        }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, UnsupportedTypeTagToInputNamesMatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "INPUT1:in1"
                input_stream: "INPUT2:in2"
                output_stream: "SUM:out"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                    tag_to_input_tensor_names {
                        key: "INPUT1"
                        value: "input1"
                    }
                    tag_to_input_tensor_names {
                        key: "INPUT2"
                        value: "input2"
                    }
                    tag_to_output_tensor_names {
                        key: "SUM"
                        value: "sum"
                    }
                    }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kOk);
}

TEST_F(OpenVINOInferenceCalculatorTest, UnsupportedTypeTagToInputNamesOutputMismatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "INPUT1:in1"
                input_stream: "INPUT2:in2"
                output_stream: "SUM:out"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                    tag_to_input_tensor_names {
                        key: "INPUT1"
                        value: "input1"
                    }
                    tag_to_input_tensor_names {
                        key: "INPUT2"
                        value: "input2"
                    }
                    tag_to_output_tensor_names {
                        key: "SUM1"
                        value: "sum"
                    }
                    }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}

TEST_F(OpenVINOInferenceCalculatorTest, UnsupportedTypeTagToInputNamesInputMismatch) {
    std::string calculator_proto =
            R"pb(
                calculator: "OpenVINOInferenceCalculator"
                input_side_packet: "SESSION:session"
                input_stream: "INPUT1:in1"
                input_stream: "INPUT2:in2"
                output_stream: "SUM:out"
                node_options: {
                    [type.googleapis.com / mediapipe.OpenVINOInferenceCalculatorOptions]: {
                    tag_to_input_tensor_names {
                        key: "INPUT3"
                        value: "input1"
                    }
                    tag_to_input_tensor_names {
                        key: "INPUT2"
                        value: "input2"
                    }
                    tag_to_output_tensor_names {
                        key: "SUM"
                        value: "sum"
                    }
                    }
                }
            )pb";
    verifyGetContract(calculator_proto, absl::StatusCode::kInternal);
}
