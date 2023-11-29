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
#include "openvinomodelserversessioncalculator.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <openvino/core/type/element_type.hpp>
#include <sstream>
#include <unordered_map>

#include "mediapipe/framework/port/gtest.h"

#include <adapters/inference_adapter.h>  // TODO fix path  model_api/model_api/cpp/adapters/include/adapters/inference_adapter.h
#include <openvino/core/shape.hpp>
#include <openvino/openvino.hpp>

#include "ovms.h"  // NOLINT
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "mediapipe/calculators/ovms/openvinomodelserversessioncalculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/deps/status_matchers.h"
//#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/tool/tag_map_helper.h"
#pragma GCC diagnostic pop
// here we need to decide if we have several calculators (1 for OVMS repository, 1-N inside mediapipe)
// for the one inside OVMS repo it makes sense to reuse code from ovms lib
using mediapipe::CalculatorContract;
using mediapipe::OpenVINOModelServerSessionCalculator;
using mediapipe::CalculatorState;
using mediapipe::ParseTextProtoOrDie;
using mediapipe::PacketType;

class OpenVINOModelServerSessionCalculatorTest : public ::testing::Test {
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

TEST_F(OpenVINOModelServerSessionCalculatorTest, VerifyCorrectPbtxtWithAllOptions) {
    auto node =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOModelServerSessionCalculator"
                output_side_packet: "SESSION:session"
                node_options: {
                  [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
                    servable_name: "not_used_name"
                    servable_version: "1"
                    server_config: "mediapipe/config.json"
                    service_url: "192.168.0.1:9178"
                  }
                }
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(node);
    auto abslStatus = mediapipe::OpenVINOModelServerSessionCalculator::GetContract(cc.get());
    ASSERT_EQ(abslStatus.code(), absl::StatusCode::kOk) << abslStatus.message();
    EXPECT_EQ(0, cc->InputSidePackets().TagMap()->NumEntries());
    EXPECT_EQ(1, cc->OutputSidePackets().NumEntries());
    auto& inputPacketsTags = cc->Inputs();
    auto& outputPacketsTags = cc->Outputs();
    EXPECT_EQ(0, inputPacketsTags.TagMap()->NumEntries());
    EXPECT_EQ(0, outputPacketsTags.TagMap()->NumEntries());
}
TEST_F(OpenVINOModelServerSessionCalculatorTest, VerifyOptionalityOfOptionFields) {
    // servable_version, server_config, server_url are optional
    auto node =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOModelServerSessionCalculator"
                output_side_packet: "SESSION:session"
                node_options: {
                  [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
                    servable_name: "not_used_name"
                  }
                }
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(node);
    auto abslStatus = mediapipe::OpenVINOModelServerSessionCalculator::GetContract(cc.get());
    EXPECT_EQ(abslStatus.code(), absl::StatusCode::kOk) << abslStatus.message();
}
TEST_F(OpenVINOModelServerSessionCalculatorTest, VerifyMandatorityOfFields) {
    // servable_version, server_config, server_url are optional
    mediapipe::CalculatorGraphConfig::Node node;
    bool success = mediapipe::ParseTextProto(
            R"pb(
                calculator: "OpenVINOModelServerSessionCalculator"
                output_side_packet: "SESSION:session"
                node_options: {
                  [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
                    # commented out servable_name: "not_used_name"
                    servable_version: "1"
                    server_config: "mediapipe/config.json"
                    service_url: "192.168.0.1:9178"
                  }
                }
            )pb", &node);
    EXPECT_FALSE(success);
}
TEST_F(OpenVINOModelServerSessionCalculatorTest, VerifyNonExistingFields) {
    mediapipe::CalculatorGraphConfig::Node node;
    bool success = mediapipe::ParseTextProto(
            R"pb(
                calculator: "OpenVINOModelServerSessionCalculator"
                output_side_packet: "SESSION:session"
                node_options: {
                  [type.googleapis.com / mediapipe.OpenVINOModelServerSessionCalculatorOptions]: {
                    servable_name: "not_used_name"
                    some_random_name: 1
                  }
                }
            )pb", &node);
    EXPECT_FALSE(success);
}
TEST_F(OpenVINOModelServerSessionCalculatorTest, MissingAllOptions) {
    auto node =
        mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig::Node>(
            R"pb(
                calculator: "OpenVINOModelServerSessionCalculator"
                output_side_packet: "SESSION:session"
            )pb");
    auto cc = absl::make_unique<CalculatorContract>();
    cc->Initialize(node);
    auto abslStatus = mediapipe::OpenVINOModelServerSessionCalculator::GetContract(cc.get());
    ASSERT_EQ(abslStatus.code(), absl::StatusCode::kInternal) << abslStatus.message();
}
