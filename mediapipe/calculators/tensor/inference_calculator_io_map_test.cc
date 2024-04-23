// Copyright 2024 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/inference_calculator_io_map.h"

#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "mediapipe/calculators/tensor/inference_calculator.pb.h"
#include "mediapipe/calculators/tensor/tensor_span.h"
#include "mediapipe/framework/api2/packet.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"

namespace mediapipe {
namespace api2 {
namespace {

constexpr char kModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "3in3out_model_swaps_input_2_and_0.tflite";

using ::mediapipe::MakePacket;
using ::mediapipe::Packet;
using ::mediapipe::tool::AddVectorSink;
using ::testing::HasSubstr;

// Defines the input/output tensor mapping and the expected order of the output
// tensors in tests.
struct InputOutputExpectedOrderTestConfig {
  std::string test_name;
  std::vector<int> input_tensor_indices_map;
  std::vector<int> output_tensor_indices_map;
  std::vector<int> expected_order;
};

static std::vector<InputOutputExpectedOrderTestConfig>
GetInputOutputExpectedOrderTestConfigs() {
  return {
      {
          .test_name = "NoRemapping",
          .input_tensor_indices_map = {},
          .output_tensor_indices_map = {},
          .expected_order = {0, 1, 2},
      },
      {
          .test_name = "InputRemappingOnly",
          .input_tensor_indices_map = {2, 1, 0},
          .output_tensor_indices_map = {},
          .expected_order = {2, 1, 0},
      },
      {
          .test_name = "OutputRemappingOnly",
          .input_tensor_indices_map = {},
          .output_tensor_indices_map = {2, 1, 0},
          .expected_order = {2, 1, 0},
      },
      {
          .test_name = "InputOutputRemapping",
          .input_tensor_indices_map = {2, 1, 0},
          .output_tensor_indices_map = {2, 1, 0},
          .expected_order = {0, 1, 2},
      },
  };
}

Tensor CreateSingleFloatTensor(float value) {
  std::vector<int> dims = {1, 1};
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape(dims));
  auto write_view = tensor.GetCpuWriteView();
  *write_view.buffer<float>() = value;
  return tensor;
}

std::vector<Tensor> CopyTensors(const TensorSpan& tensors) {
  std::vector<Tensor> result;
  result.reserve(tensors.size());
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    result.emplace_back(tensor.element_type(), tensor.shape());
    const auto read_view = tensor.GetCpuReadView();
    auto write_view = result.back().GetCpuWriteView();
    memcpy(write_view.buffer<void>(), read_view.buffer<void>(), tensor.bytes());
  }
  return result;
}

InferenceCalculatorOptions::InputOutputConfig GenerateInputOutputMap(
    const InputOutputExpectedOrderTestConfig& config) {
  InferenceCalculatorOptions::InputOutputConfig result;
  for (const int index : config.input_tensor_indices_map) {
    result.mutable_input_tensor_indices_map()->add_model_tensor_indices(index);
  }
  for (const int index : config.output_tensor_indices_map) {
    result.mutable_output_tensor_indices_map()->add_model_tensor_indices(index);
  }
  return result;
}

using InferenceCalculatorIoMapTestWithParams =
    testing::TestWithParam<InputOutputExpectedOrderTestConfig>;
TEST_P(InferenceCalculatorIoMapTestWithParams,
       ShouldRemapInputAndOutputTensors) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  const InputOutputExpectedOrderTestConfig& config = GetParam();
  const InferenceCalculatorOptions::InputOutputConfig map =
      GenerateInputOutputMap(config);

  MP_ASSERT_OK_AND_ASSIGN(
      auto mapped_input_tensors,
      RemapInputTensors(MakeTensorSpan(input_tensors_unmapped), map));
  for (int i = 0; i < config.input_tensor_indices_map.size(); ++i) {
    EXPECT_FLOAT_EQ(mapped_input_tensors[i].GetCpuReadView().buffer<float>()[0],
                    config.input_tensor_indices_map[i]);
  }

  MP_ASSERT_OK_AND_ASSIGN(
      auto mapped_output_tensors,
      RemapOutputTensors(CopyTensors(mapped_input_tensors), map));

  for (int i = 0; i < kNumTensors; ++i) {
    EXPECT_FLOAT_EQ(
        mapped_output_tensors[i].GetCpuReadView().buffer<float>()[0],
        config.expected_order[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    InferenceCalculatorIoMapTestSuiteInitialization,
    InferenceCalculatorIoMapTestWithParams,  // This is the name of your
                                             // parameterized test
    testing::ValuesIn(GetInputOutputExpectedOrderTestConfigs()),
    [](const testing::TestParamInfo<
        InferenceCalculatorIoMapTestWithParams::ParamType>& info) {
      return info.param.test_name;
    });

TEST(InferenceCalculatorIoMapTest, ShouldReportOutOfBoundsInputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 100,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(RemapInputTensors(MakeTensorSpan(input_tensors_unmapped), map),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Index 100 out of range")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportOutOfBoundsOutputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        output_tensor_indices_map {
          model_tensor_indices: 100,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(RemapOutputTensors(std::move(output_tensors_unmapped), map),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Index 100 out of range")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportTooFewInputMappingIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(
      RemapInputTensors(MakeTensorSpan(input_tensors_unmapped), map),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Number of input tensors does not match the size of "
                         "model_tensor_indices list")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportTooFewOutputMappingIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        output_tensor_indices_map {
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(RemapOutputTensors(std::move(output_tensors_unmapped), map),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Number of output tensors does not match the "
                                 "size of model_tensor_indices list")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportTooManyMappingInputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 3,
          model_tensor_indices: 2,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(
      RemapInputTensors(MakeTensorSpan(input_tensors_unmapped), map),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Number of input tensors does not match the size of "
                         "model_tensor_indices list")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportTooManyMappingOutputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.emplace_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        output_tensor_indices_map {
          model_tensor_indices: 3,
          model_tensor_indices: 2,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  EXPECT_THAT(
      RemapOutputTensors(std::move(output_tensors_unmapped), map),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Number of output tensors does not match the size of "
                         "model_tensor_indices list")));
}

TEST(InferenceCalculatorIoMapTest, ShouldReportDuplicatedMappingIndices) {
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 2,
          model_tensor_indices: 2,
          model_tensor_indices: 1
        }
      )pb");

  EXPECT_THAT(
      VerifyInputOutputConfig(map),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Indices in TensorIndicesMap are not unique.")));
}

class InferenceCalculatorIoMapSmokeTest
    : public ::testing::TestWithParam<InputOutputExpectedOrderTestConfig> {
 protected:
  void SetUpGraphAndRun(
      const InferenceCalculatorOptions::InputOutputConfig& io_config,
      const std::vector<int>& expected_order,
      bool pass_config_as_side_packet = false) {
    CalculatorGraph graph;
    CalculatorGraphConfig graph_config =
        ParseTextProtoOrDie<CalculatorGraphConfig>(absl::StrReplaceAll(
            R"pb(
              input_stream: "input0"
              input_stream: "input1"
              input_stream: "input2"
              output_stream: "output0"
              output_stream: "output1"
              output_stream: "output2"

              node {
                calculator: "InferenceCalculator"
                # Note that the test model swaps tensor 2 and tensor 0, see
                # signature below:
                #~~~~~~~~~~ INPUTS ~~~~~~~~~~
                # 0 :  input_0 :  [1 1] :  F32
                # 1 :  input_1 :  [1 1] :  F32
                # 2 :  input_2 :  [1 1] :  F32
                #~~~~~~~~~~ OUTPUTS ~~~~~~~~~
                # 0 :  output_2 :  [1 1] :  F32
                # 1 :  output_1 :  [1 1] :  F32
                # 2 :  output_0 :  [1 1] :  F32
                # To compensate for this, the output streams 0 & 1 are
                # swapped.
                input_stream: "TENSOR:0:input0"
                input_stream: "TENSOR:1:input1"
                input_stream: "TENSOR:2:input2"
                output_stream: "TENSOR:0:output2"
                output_stream: "TENSOR:1:output1"
                output_stream: "TENSOR:2:output0"
                options {
                  [mediapipe.InferenceCalculatorOptions.ext] {
                    model_path: "$model"
                    delegate {}  # empty delegate message enables CPU inference.
                  }
                }
              }
            )pb",
            {{"$model", kModelPath}}));

    if (pass_config_as_side_packet) {
      *graph_config.mutable_node(0)->mutable_input_side_packet()->Add() =
          "IO_CONFIG:io_config";
    } else {
      *graph_config.mutable_node(0)
           ->mutable_options()
           ->MutableExtension(InferenceCalculatorOptions::ext)
           ->mutable_input_output_config() = io_config;
    }

    std::vector<std::vector<Packet>> output_packets(3);
    AddVectorSink("output0", &graph_config, &output_packets[0]);
    AddVectorSink("output1", &graph_config, &output_packets[1]);
    AddVectorSink("output2", &graph_config, &output_packets[2]);

    MP_EXPECT_OK(graph.Initialize(graph_config));

    std::map<std::string, Packet> side_packets;
    if (pass_config_as_side_packet) {
      side_packets["io_config"] =
          MakePacket<InferenceCalculatorOptions::InputOutputConfig>(io_config);
    }
    MP_EXPECT_OK(graph.StartRun(side_packets));
    for (int n = 0; n < 3; ++n) {
      Tensor input_tensor = CreateSingleFloatTensor(n);
      MP_EXPECT_OK(graph.AddPacketToInputStream(
          absl::StrCat("input", n),
          MakePacket<Tensor>(std::move(input_tensor)).At(Timestamp(0))));
    }
    MP_EXPECT_OK(graph.WaitUntilIdle());

    for (int i = 0; i < output_packets.size(); ++i) {
      EXPECT_EQ(output_packets[i].size(), 1);
      const auto read_view =
          output_packets[i][0].Get<Tensor>().GetCpuReadView();
      // Tensor float value should match the index in expected_order.
      EXPECT_FLOAT_EQ(read_view.buffer<float>()[0], expected_order[i]);
    }
  }

  std::string model_path_;
};

TEST_P(InferenceCalculatorIoMapSmokeTest, SmokeTestWithIoMapConfig) {
  const InputOutputExpectedOrderTestConfig& params = GetParam();
  InferenceCalculatorOptions::InputOutputConfig io_config;
  for (const int index : params.input_tensor_indices_map) {
    io_config.mutable_input_tensor_indices_map()->add_model_tensor_indices(
        index);
  }
  for (const int index : params.output_tensor_indices_map) {
    io_config.mutable_output_tensor_indices_map()->add_model_tensor_indices(
        index);
  }
  SetUpGraphAndRun(io_config, params.expected_order);
}

TEST_P(InferenceCalculatorIoMapSmokeTest, SmokeTestWithIoMapSidePacket) {
  const InputOutputExpectedOrderTestConfig& params = GetParam();
  InferenceCalculatorOptions::InputOutputConfig io_config;
  for (const int index : params.input_tensor_indices_map) {
    io_config.mutable_input_tensor_indices_map()->add_model_tensor_indices(
        index);
  }
  for (const int index : params.output_tensor_indices_map) {
    io_config.mutable_output_tensor_indices_map()->add_model_tensor_indices(
        index);
  }
  SetUpGraphAndRun(io_config, params.expected_order,
                   /*pass_config_as_side_packet=*/true);
}

INSTANTIATE_TEST_SUITE_P(
    InferenceCalculatorIoMapSmokeParamTest, InferenceCalculatorIoMapSmokeTest,
    ::testing::ValuesIn(GetInputOutputExpectedOrderTestConfigs()));

}  // namespace
}  // namespace api2
}  // namespace mediapipe
