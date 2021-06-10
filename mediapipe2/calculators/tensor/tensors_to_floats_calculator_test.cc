// Copyright 2020 The MediaPipe Authors.
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

#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensor/tensors_to_floats_calculator.pb.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {

using mediapipe::ParseTextProtoOrDie;
using Node = ::mediapipe::CalculatorGraphConfig::Node;

const float kErrorMargin = 1e-2f;

class TensorsToFloatsCalculatorTest : public ::testing::Test {
 protected:
  void BuildGraph(mediapipe::CalculatorRunner* runner,
                  const std::vector<float>& values) {
    auto tensors = absl::make_unique<std::vector<Tensor>>();
    tensors->emplace_back(
        Tensor::ElementType::kFloat32,
        Tensor::Shape{1, 1, static_cast<int>(values.size()), 1});
    auto view = tensors->back().GetCpuWriteView();
    float* tensor_buffer = view.buffer<float>();
    ASSERT_NE(tensor_buffer, nullptr);
    for (int i = 0; i < values.size(); ++i) {
      tensor_buffer[i] = values[i];
    }

    int64 stream_timestamp = 0;
    auto& input_stream_packets =
        runner->MutableInputs()->Tag("TENSORS").packets;

    input_stream_packets.push_back(
        mediapipe::Adopt(tensors.release())
            .At(mediapipe::Timestamp(stream_timestamp++)));
  }
};

TEST_F(TensorsToFloatsCalculatorTest, SingleValue) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToFloatsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "FLOAT:float"
  )pb"));

  const float single_value = 0.5;
  BuildGraph(&runner, {single_value});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("FLOAT").packets;

  EXPECT_EQ(1, output_packets_.size());

  const auto& value = output_packets_[0].Get<float>();
  EXPECT_EQ(single_value, value);
}

TEST_F(TensorsToFloatsCalculatorTest, SingleValueAsVector) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToFloatsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "FLOATS:floats"
  )pb"));

  const float single_value = 0.5;
  BuildGraph(&runner, {single_value});
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("FLOATS").packets;
  EXPECT_EQ(1, output_packets_.size());

  const auto& values = output_packets_[0].Get<std::vector<float>>();
  EXPECT_EQ(1, values.size());
  EXPECT_EQ(single_value, values[0]);
}

TEST_F(TensorsToFloatsCalculatorTest, FloatVector) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToFloatsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "FLOATS:floats"
  )pb"));

  const std::vector<float> input_values = {0.f, 0.5f, 1.0f};
  BuildGraph(&runner, input_values);
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("FLOATS").packets;
  EXPECT_EQ(1, output_packets_.size());

  const auto& values = output_packets_[0].Get<std::vector<float>>();
  EXPECT_EQ(input_values.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(values[i], input_values[i], kErrorMargin);
  }
}

TEST_F(TensorsToFloatsCalculatorTest, FloatVectorWithSigmoid) {
  mediapipe::CalculatorRunner runner(ParseTextProtoOrDie<Node>(R"pb(
    calculator: "TensorsToFloatsCalculator"
    input_stream: "TENSORS:tensors"
    output_stream: "FLOATS:floats"
    options {
      [mediapipe.TensorsToFloatsCalculatorOptions.ext] { activation: SIGMOID }
    }
  )pb"));

  const std::vector<float> input_values = {-1.f, 0.f, 1.0f};
  const std::vector<float> expected_output_with_sigmoid = {0.269f, 0.5f,
                                                           0.731f};
  BuildGraph(&runner, input_values);
  MP_ASSERT_OK(runner.Run());

  const auto& output_packets_ = runner.Outputs().Tag("FLOATS").packets;
  EXPECT_EQ(1, output_packets_.size());

  const auto& values = output_packets_[0].Get<std::vector<float>>();
  EXPECT_EQ(expected_output_with_sigmoid.size(), values.size());
  for (int i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(values[i], expected_output_with_sigmoid[i], kErrorMargin);
  }
}

}  // namespace mediapipe
