// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensorflow/vector_float_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class VectorToTensorFloatCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(
      const VectorFloatToTensorCalculatorOptions::InputSize input_size,
      const bool transpose) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("VectorFloatToTensorCalculator");
    config.add_input_stream("input_float");
    config.add_output_stream("output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        VectorFloatToTensorCalculatorOptions::ext);
    options->set_input_size(input_size);
    options->set_transpose(transpose);
    runner_ = ::absl::make_unique<CalculatorRunner>(config);
  }

  void TestConvertFromVectoVectorFloat(const bool transpose) {
    SetUpRunner(VectorFloatToTensorCalculatorOptions::INPUT_2D, transpose);
    auto input = ::absl::make_unique<std::vector<std::vector<float>>>(
        2, std::vector<float>(2));
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        // 2^i can be represented exactly in floating point numbers
        // if 'i' is small.
        input->at(i).at(j) = static_cast<float>(1 << (i * 2 + j));
      }
    }

    const int64 time = 1234;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(time)));

    EXPECT_TRUE(runner_->Run().ok());

    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    EXPECT_EQ(1, output_packets.size());
    EXPECT_EQ(time, output_packets[0].Timestamp().Value());
    const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

    EXPECT_EQ(2, output_tensor.dims());
    EXPECT_EQ(tf::DT_FLOAT, output_tensor.dtype());
    const auto matrix = output_tensor.matrix<float>();

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (!transpose) {
          EXPECT_EQ(1 << (i * 2 + j), matrix(i, j));
        } else {
          EXPECT_EQ(1 << (j * 2 + i), matrix(i, j));
        }
      }
    }
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(VectorToTensorFloatCalculatorTest, ConvertsFromVectorFloat) {
  SetUpRunner(VectorFloatToTensorCalculatorOptions::INPUT_1D, false);
  auto input = ::absl::make_unique<std::vector<float>>(5);
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    input->at(i) = static_cast<float>(1 << i);
  }
  const int64 time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(input.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(1, output_tensor.dims());
  EXPECT_EQ(tf::DT_FLOAT, output_tensor.dtype());
  const auto vec = output_tensor.vec<float>();

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(1 << i, vec(i));
  }
}

TEST_F(VectorToTensorFloatCalculatorTest, ConvertsFromVectorVectorFloat) {
  for (bool transpose : {false, true}) {
    TestConvertFromVectoVectorFloat(transpose);
  }
}

}  // namespace
}  // namespace mediapipe
