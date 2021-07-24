// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/calculator_contract.h"

// TODO: Move protos in another CL after the C++ code migration.
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_contract_test.pb.h"
#include "mediapipe/framework/packet_generator.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/status_handler.pb.h"

namespace mediapipe {

namespace {

TEST(CalculatorContractTest, Calculator) {
  const CalculatorGraphConfig::Node node =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "MixtureOfExpertsFusionCalculator"
        input_stream: "FRAME:fdense_pca_moe_aggregated_detection"
        input_stream: "FNET:fnet_logreg_aggregated_detection"
        input_stream: "EGRAPH:egraph_segment_aggregated_detection"
        input_stream: "VIDEO:fdense_averaged_pca_moe_v2_detection"
        input_side_packet: "FUSION_MODEL:egraph_topical_packet_factory"
        output_stream: "egraph_topical_detection"
      )pb");
  CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node));
  EXPECT_EQ(contract.Inputs().NumEntries(), 4);
  EXPECT_EQ(contract.Outputs().NumEntries(), 1);
  EXPECT_EQ(contract.InputSidePackets().NumEntries(), 1);
  EXPECT_EQ(contract.OutputSidePackets().NumEntries(), 0);
}

TEST(CalculatorContractTest, CalculatorOptions) {
  const CalculatorGraphConfig::Node node =
      mediapipe::ParseTextProtoOrDie<CalculatorGraphConfig::Node>(R"pb(
        calculator: "CalculatorTestCalculator"
        input_stream: "DATA:ycbcr_frames"
        input_stream: "VIDEO_HEADER:ycbcr_frames_prestream"
        output_stream: "DATA:ycbcr_downsampled"
        output_stream: "VIDEO_HEADER:ycbcr_downsampled_prestream"
        options {
          [mediapipe.CalculatorContractTestOptions.ext] { test_field: 1.0 }
        })pb");
  CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node));
  const auto& test_options =
      contract.Options().GetExtension(CalculatorContractTestOptions::ext);
  EXPECT_EQ(test_options.test_field(), 1.0);
  EXPECT_EQ(contract.Inputs().NumEntries(), 2);
  EXPECT_EQ(contract.Outputs().NumEntries(), 2);
  EXPECT_EQ(contract.InputSidePackets().NumEntries(), 0);
  EXPECT_EQ(contract.OutputSidePackets().NumEntries(), 0);
}

TEST(CalculatorContractTest, PacketGenerator) {
  const PacketGeneratorConfig node =
      mediapipe::ParseTextProtoOrDie<PacketGeneratorConfig>(R"pb(
        packet_generator: "DaredevilLabeledTimeSeriesGenerator"
        input_side_packet: "labeled_time_series"
        output_side_packet: "time_series_header"
        output_side_packet: "input_matrix"
        output_side_packet: "label_set"
        output_side_packet: "content_fingerprint"
      )pb");
  CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node, ""));
  EXPECT_EQ(contract.InputSidePackets().NumEntries(), 1);
  EXPECT_EQ(contract.OutputSidePackets().NumEntries(), 4);
}

TEST(CalculatorContractTest, StatusHandler) {
  const StatusHandlerConfig node =
      mediapipe::ParseTextProtoOrDie<StatusHandlerConfig>(R"pb(
        status_handler: "TaskInjectorStatusHandler"
        input_side_packet: "ROW:cid"
        input_side_packet: "SPEC:task_specification"
      )pb");
  CalculatorContract contract;
  MP_EXPECT_OK(contract.Initialize(node));
  EXPECT_EQ(contract.InputSidePackets().NumEntries(), 2);
}

}  // namespace
}  // namespace mediapipe
