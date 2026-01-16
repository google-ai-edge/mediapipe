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

#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/tensorflow/vector_string_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class VectorStringToTensorCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(
      const VectorStringToTensorCalculatorOptions::InputSize input_size,
      const bool transpose) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("VectorStringToTensorCalculator");
    config.add_input_stream("input_string");
    config.add_output_stream("output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        VectorStringToTensorCalculatorOptions::ext);
    options->set_input_size(input_size);
    options->set_transpose(transpose);
    runner_ = ::absl::make_unique<CalculatorRunner>(config);
  }

  void TestConvertFromVectoVectorString(const bool transpose) {
    SetUpRunner(VectorStringToTensorCalculatorOptions::INPUT_2D, transpose);
    auto input = ::absl::make_unique<std::vector<std::vector<std::string>>>(
        2, std::vector<std::string>(2));
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        input->at(i).at(j) = absl::StrCat(i, j);
      }
    }

    const int64_t time = 1234;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(time)));

    EXPECT_TRUE(runner_->Run().ok());

    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    EXPECT_EQ(1, output_packets.size());
    EXPECT_EQ(time, output_packets[0].Timestamp().Value());
    const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

    EXPECT_EQ(2, output_tensor.dims());
    EXPECT_EQ(tf::DT_STRING, output_tensor.dtype());
    const auto matrix = output_tensor.matrix<tf::tstring>();

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        if (!transpose) {
          EXPECT_EQ(absl::StrCat(i, j), matrix(i, j));
        } else {
          EXPECT_EQ(absl::StrCat(j, i), matrix(i, j));
        }
      }
    }
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(VectorStringToTensorCalculatorTest, ConvertsFromVectorString) {
  SetUpRunner(VectorStringToTensorCalculatorOptions::INPUT_1D, false);
  auto input = ::absl::make_unique<std::vector<std::string>>(5);
  for (int i = 0; i < 5; ++i) {
    input->at(i) = absl::StrCat(i);
  }
  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(input.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();

  EXPECT_EQ(1, output_tensor.dims());
  EXPECT_EQ(tf::DT_STRING, output_tensor.dtype());
  const auto vec = output_tensor.vec<tf::tstring>();

  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(absl::StrCat(i), vec(i));
  }
}

TEST_F(VectorStringToTensorCalculatorTest, ConvertsFromVectorVectorString) {
  for (bool transpose : {false, true}) {
    TestConvertFromVectoVectorString(transpose);
  }
}

}  // namespace
}  // namespace mediapipe
