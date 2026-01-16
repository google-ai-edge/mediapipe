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

#include "mediapipe/calculators/tensor/inference_io_mapper.h"

#include <cstring>
#include <map>
#include <memory>
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
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/resources.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/util/tflite/tflite_model_loader.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

namespace mediapipe {
namespace api2 {
namespace {

// Signature of 3in3out_model_swaps_input_2_and_0.tflite model:
// ~~~~~~~~~~ INPUTS ~~~~~~~~~~
// 0 :  third_input :  [1 1] :  F32
// 1 :  first_input :  [1 1] :  F32
// 2 :  second_input :  [1 1] :  F32
// ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
// 0 :  output_1 :  [1 1] :  F32
// 1 :  output_0 :  [1 1] :  F32
// 2 :  output_2 :  [1 1] :  F32
constexpr char k3In3OutSwaps2And0ModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "3in3out_model_swaps_input_2_and_0.tflite";

// Model contains two signatures.
constexpr char kTwoSignaturesModelPath[] =
    "mediapipe/calculators/tensor/testdata/"
    "test_two_signature_keys_model.tflite";

using ::mediapipe::MakePacket;
using ::mediapipe::Packet;
using ::mediapipe::tool::AddVectorSink;
using ::testing::HasSubstr;
using ::tflite::impl::InterpreterBuilder;
using ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates;

// Defines the input/output tensor mapping and the expected order of the output
// tensors in tests.
struct InputOutputExpectedOrderTestConfig {
  std::string test_name;
  std::vector<int> input_tensor_indices_map;
  std::vector<std::string> input_tensor_names_map;
  std::vector<int> output_tensor_indices_map;
  std::vector<std::string> output_tensor_names_map;
  std::vector<int> expected_test_value_order;
};

static std::vector<InputOutputExpectedOrderTestConfig>
GetInputOutputExpectedOrderTestConfigs() {
  return {
      // All tests populate the three InferenceCalculator input tensors with
      // the values 0, 1, 2.
      {
          .test_name = "NoRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 0
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 2
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 2
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 0
          .expected_test_value_order = {2, 1, 0},
      },
      {
          .test_name = "InputIndicesRemapping",
          .input_tensor_indices_map = {2, 1, 0},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ REMAPPED INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 2
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 0
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 0
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 2
          .expected_test_value_order = {0, 1, 2},
      },
      {
          .test_name = "OutputIndicesRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {2, 1, 0},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 0
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 2
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 2
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 0
          // ~~~~~~~~~~ REMAPPED OUTPUTS ~~~~~~~~~~
          // 0 :  output_2 :  [1 1] :  F32      // Output value 0
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_1 :  [1 1] :  F32      // Output value 2
          .expected_test_value_order = {0, 1, 2},
      },
      {
          .test_name = "InputOutputIndicesRemapping",
          .input_tensor_indices_map = {2, 1, 0},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {2, 1, 0},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 2
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 0
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 0
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 2
          // ~~~~~~~~~~ REMAPPED OUTPUTS ~~~~~~~~~~
          // 0 :  output_2 :  [1 1] :  F32      // Output value 2
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_1 :  [1 1] :  F32      // Output value 0
          .expected_test_value_order = {2, 1, 0},
      },
      {
          .test_name = "InputNameBasedRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map = {"first_input",   // Input test value 0
                                     "second_input",  // Input test value 1
                                     "third_input"},  // Input test value 2
          .output_tensor_indices_map = {},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 2
          // 1 :  first_input :  [1 1] :  F32   // Input value 0
          // 2 :  second_input :  [1 1] :  F32  // Input value 1
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 1
          // 1 :  output_0 :  [1 1] :  F32      // Output value 0
          // 2 :  output_2 :  [1 1] :  F32      // Output value 2
          .expected_test_value_order = {1, 0, 2},
      },
      {
          .test_name = "RotatedInputNameBasedRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map =
              {
                  "second_input",  // Input value 0
                  "third_input",   // Input value 1
                  "first_input",   // Input value 2
              },
          .output_tensor_indices_map = {},
          .output_tensor_names_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 1
          // 1 :  first_input :  [1 1] :  F32   // Input value 2
          // 2 :  second_input :  [1 1] :  F32  // Input value 0
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 0
          // 1 :  output_0 :  [1 1] :  F32      // Output value 2
          // 2 :  output_2 :  [1 1] :  F32      // Output value 1
          .expected_test_value_order =
              {0, 2, 1},  // Rotated input order compared to above.
      },
      {
          .test_name = "OutputNameBasedRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 0
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 2
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 2
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 0
          // ~~~~~~~~~~ REMAPPED OUTPUTS ~~~~~~~~~~
          // 0 :  output_0 :  [1 1] :  F32      // Output value 1
          // 1 :  output_1 :  [1 1] :  F32      // Output value 2
          // 2 :  output_2 :  [1 1] :  F32      // Output value 0
          .output_tensor_names_map = {"output_0", "output_1", "output_2"},
          .expected_test_value_order = {1, 2, 0},
      },
      {
          .test_name = "RotatedOutputNameBasedRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map = {},
          .output_tensor_indices_map = {},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 0
          // 1 :  first_input :  [1 1] :  F32   // Input value 1
          // 2 :  second_input :  [1 1] :  F32  // Input value 2
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 2
          // 1 :  output_0 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 0
          // ~~~~~~~~~~ REMAPPED OUTPUTS ~~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 2
          // 1 :  output_2 :  [1 1] :  F32      // Output value 0
          // 2 :  output_0 :  [1 1] :  F32      // Output value 1
          .output_tensor_names_map = {"output_1", "output_2", "output_0"},
          .expected_test_value_order = {2, 0, 1},
      },
      {
          .test_name = "InputAndOutputNameBasedRemapping",
          .input_tensor_indices_map = {},
          .input_tensor_names_map =
              {
                  "first_input",   // Input value 0
                  "second_input",  // Input value 1
                  "third_input",   // Input value 2
              },
          .output_tensor_indices_map = {},
          .output_tensor_names_map = {"output_0", "output_1", "output_2"},
          // ~~~~~~~~~~ INPUTS ~~~~~~~~~~
          // 0 :  third_input :  [1 1] :  F32   // Input value 2
          // 1 :  first_input :  [1 1] :  F32   // Input value 0
          // 2 :  second_input :  [1 1] :  F32  // Input value 1
          // ~~~~~~~~~~ OUTPUTS ~~~~~~~~~
          // 0 :  output_1 :  [1 1] :  F32      // Output value 1
          // 1 :  output_0 :  [1 1] :  F32      // Output value 0
          // 2 :  output_2 :  [1 1] :  F32      // Output value 2
          // ~~~~~~~~~~ REMAPPED OUTPUTS ~~~~~~~~~~
          // 0 :  output_0 :  [1 1] :  F32      // Output value 0
          // 1 :  output_1 :  [1 1] :  F32      // Output value 1
          // 2 :  output_2 :  [1 1] :  F32      // Output value 2
          .expected_test_value_order = {0, 1, 2},
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

std::vector<Tensor> SimulateInference(const TensorSpan& tensors) {
  std::vector<Tensor> result;
  // Simulate tensor swap by 3in3out_model_swaps_input_2_and_0.tflite
  const std::vector<int> tensor_mapping = {2, 1, 0};
  result.reserve(tensors.size());
  for (int i = 0; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[tensor_mapping[i]];
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
  for (const std::string& name : config.input_tensor_names_map) {
    result.mutable_input_tensor_names_map()->add_tensor_names(name);
  }
  for (const int index : config.output_tensor_indices_map) {
    result.mutable_output_tensor_indices_map()->add_model_tensor_indices(index);
  }
  for (const std::string& name : config.output_tensor_names_map) {
    result.mutable_output_tensor_names_map()->add_tensor_names(name);
  }
  return result;
}

class InferenceCalculatorIoMapTestWithParams
    : public ::testing::TestWithParam<InputOutputExpectedOrderTestConfig> {
 protected:
  void SetUp() override {
    std::unique_ptr<Resources> resources = CreateDefaultResources();
    MP_ASSERT_OK_AND_ASSIGN(
        model_, TfLiteModelLoader::LoadFromPath(*resources,
                                                k3In3OutSwaps2And0ModelPath));
    InterpreterBuilder(
        *model_.Get(),
        BuiltinOpResolverWithoutDefaultDelegates())(&interpreter_);
  }

  api2::Packet<TfLiteModelPtr> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

TEST_P(InferenceCalculatorIoMapTestWithParams,
       ShouldRemapInputAndOutputTensors) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  const InputOutputExpectedOrderTestConfig& config = GetParam();
  const InferenceCalculatorOptions::InputOutputConfig map =
      GenerateInputOutputMap(config);

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));

  MP_ASSERT_OK_AND_ASSIGN(
      auto mapped_input_tensors,
      mapper.RemapInputTensors(MakeTensorSpan(input_tensors_unmapped)));
  for (int i = 0; i < config.input_tensor_indices_map.size(); ++i) {
    EXPECT_FLOAT_EQ(mapped_input_tensors[i].GetCpuReadView().buffer<float>()[0],
                    config.input_tensor_indices_map[i]);
  }

  MP_ASSERT_OK_AND_ASSIGN(
      auto mapped_output_tensors,
      mapper.RemapOutputTensors(SimulateInference(mapped_input_tensors)));

  for (int i = 0; i < kNumTensors; ++i) {
    EXPECT_FLOAT_EQ(
        mapped_output_tensors[i].GetCpuReadView().buffer<float>()[0],
        config.expected_test_value_order[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    InferenceCalculatorIoMapTestSuiteInitialization,
    InferenceCalculatorIoMapTestWithParams,
    testing::ValuesIn(GetInputOutputExpectedOrderTestConfigs()),
    [](const testing::TestParamInfo<
        InferenceCalculatorIoMapTestWithParams::ParamType>& info) {
      return info.param.test_name;
    });

class InferenceIoMapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::unique_ptr<Resources> resources = CreateDefaultResources();
    MP_ASSERT_OK_AND_ASSIGN(
        model_, TfLiteModelLoader::LoadFromPath(*resources,
                                                k3In3OutSwaps2And0ModelPath));
    InterpreterBuilder(
        *model_.Get(),
        BuiltinOpResolverWithoutDefaultDelegates())(&interpreter_);
  }

  api2::Packet<TfLiteModelPtr> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
};

TEST_F(InferenceIoMapperTest, ShouldReportOutOfBoundsInputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 100,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapInputTensors(MakeTensorSpan(input_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Index 100 out of range")));
}

TEST_F(InferenceIoMapperTest, ShouldReportOutOfBoundsOutputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        output_tensor_indices_map {
          model_tensor_indices: 100,
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapOutputTensors(std::move(output_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Index 100 out of range")));
}

TEST_F(InferenceIoMapperTest, ShouldReportTooFewInputMappingIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapInputTensors(MakeTensorSpan(input_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unexpected number of input tensors")));
}

TEST_F(InferenceIoMapperTest, ShouldReportTooFewOutputMappingIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        output_tensor_indices_map {
          model_tensor_indices: 1,
          model_tensor_indices: 0
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapOutputTensors(std::move(output_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unexpected number of output tensors")));
}

TEST_F(InferenceIoMapperTest, ShouldReportTooManyMappingInputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
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

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapInputTensors(MakeTensorSpan(input_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unexpected number of input tensors")));
}

TEST_F(InferenceIoMapperTest, ShouldReportTooManyMappingOutputIndices) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> output_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    output_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
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

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
  EXPECT_THAT(mapper.RemapOutputTensors(std::move(output_tensors_unmapped)),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unexpected number of output tensors")));
}

TEST_F(InferenceIoMapperTest, ShouldReportDuplicatedMappingIndices) {
  InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_indices_map {
          model_tensor_indices: 2,
          model_tensor_indices: 2,
          model_tensor_indices: 1
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(
      mapper.UpdateIoMap(map, input_output_tensor_names),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Indices in TensorIndicesMap are not unique.")));
}

TEST_F(InferenceIoMapperTest, ShouldDetectDuplicatedTensorNames) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  const InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map {
          tensor_names: "first_input",
          tensor_names: "first_input",
          tensor_names: "third_input"
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(mapper.UpdateIoMap(map, input_output_tensor_names),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Duplicate tensor names found")));
}

TEST_F(InferenceIoMapperTest, ShouldDetectNoneExistingTensorNames) {
  constexpr int kNumTensors = 3;
  std::vector<Tensor> input_tensors_unmapped;
  for (int i = 0; i < kNumTensors; ++i) {
    input_tensors_unmapped.push_back(CreateSingleFloatTensor(i));
  }
  const InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map {
          tensor_names: "abc",
          tensor_names: "first_input",
          tensor_names: "third_input"
        }
      )pb");

  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter_));
  EXPECT_THAT(mapper.UpdateIoMap(map, input_output_tensor_names),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Tensor name abc not found")));
}

class InferenceIoMapperSmokeTest
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
                input_stream: "TENSOR:0:input0"
                input_stream: "TENSOR:1:input1"
                input_stream: "TENSOR:2:input2"
                output_stream: "TENSOR:0:output0"
                output_stream: "TENSOR:1:output1"
                output_stream: "TENSOR:2:output2"
                options {
                  [mediapipe.InferenceCalculatorOptions.ext] {
                    model_path: "$model"
                    delegate {}  # empty delegate message enables CPU inference.
                  }
                }
              }
            )pb",
            {{"$model", k3In3OutSwaps2And0ModelPath}}));

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
    MP_EXPECT_OK(graph.CloseAllInputStreams());
    MP_EXPECT_OK(graph.WaitUntilDone());

    EXPECT_EQ(output_packets.size(), expected_order.size());
    for (int i = 0; i < output_packets.size(); ++i) {
      EXPECT_EQ(output_packets[i].size(), 1);
      const auto read_view =
          output_packets[i][0].Get<Tensor>().GetCpuReadView();
      EXPECT_FLOAT_EQ(read_view.buffer<float>()[0], expected_order[i]);
    }
  }

  std::string model_path_;
};

TEST_P(InferenceIoMapperSmokeTest, SmokeTestWithIoMapConfig) {
  const InputOutputExpectedOrderTestConfig& config = GetParam();
  const InferenceCalculatorOptions::InputOutputConfig io_map =
      GenerateInputOutputMap(config);
  SetUpGraphAndRun(io_map, config.expected_test_value_order);
}

TEST_P(InferenceIoMapperSmokeTest, SmokeTestWithIoMapSidePacket) {
  const InputOutputExpectedOrderTestConfig& config = GetParam();
  const InferenceCalculatorOptions::InputOutputConfig io_map =
      GenerateInputOutputMap(config);
  SetUpGraphAndRun(io_map, config.expected_test_value_order,
                   /*pass_config_as_side_packet=*/true);
}

INSTANTIATE_TEST_SUITE_P(
    InferenceCalculatorIoMapSmokeParamTest, InferenceIoMapperSmokeTest,
    ::testing::ValuesIn(GetInputOutputExpectedOrderTestConfigs()),
    [](const testing::TestParamInfo<InferenceIoMapperSmokeTest::ParamType>&
           info) { return info.param.test_name; });

TEST(InferenceIoMapper,
     ShouldIgnoreMultiSignatureChecksWhenNoNameBasedMapConfigExists) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();
  MP_ASSERT_OK_AND_ASSIGN(
      const auto model,
      TfLiteModelLoader::LoadFromPath(*resources, kTwoSignaturesModelPath));
  std::unique_ptr<tflite::Interpreter> interpreter;
  InterpreterBuilder(*model.Get(),
                     BuiltinOpResolverWithoutDefaultDelegates())(&interpreter);

  InferenceCalculatorOptions::InputOutputConfig map;
  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter));
  MP_EXPECT_OK(mapper.UpdateIoMap(map, input_output_tensor_names));
}

TEST(InferenceIoMapper, ShouldFailWhenMultipleSignaturesExist) {
  std::unique_ptr<Resources> resources = CreateDefaultResources();
  MP_ASSERT_OK_AND_ASSIGN(
      const auto model,
      TfLiteModelLoader::LoadFromPath(*resources, kTwoSignaturesModelPath));
  std::unique_ptr<tflite::Interpreter> interpreter;
  InterpreterBuilder(*model.Get(),
                     BuiltinOpResolverWithoutDefaultDelegates())(&interpreter);

  const InferenceCalculatorOptions::InputOutputConfig map =
      ParseTextProtoOrDie<InferenceCalculatorOptions::InputOutputConfig>(R"pb(
        input_tensor_names_map {
          tensor_names: "abc",
        }
      )pb");
  InferenceIoMapper mapper;
  MP_ASSERT_OK_AND_ASSIGN(
      const auto input_output_tensor_names,
      InferenceIoMapper::GetInputOutputTensorNamesFromInterpreter(
          *interpreter));
  EXPECT_THAT(mapper.UpdateIoMap(map, input_output_tensor_names),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("not supported with multi-signature models")));
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
