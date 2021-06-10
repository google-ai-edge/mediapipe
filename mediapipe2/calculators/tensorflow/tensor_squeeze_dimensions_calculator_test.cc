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

#include "mediapipe/calculators/tensorflow/tensor_squeeze_dimensions_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class TensorSqueezeDimensionsCalculatorTest : public ::testing::Test {
 protected:
  TensorSqueezeDimensionsCalculatorTest() {
    // Initialize tensor_ with deterministic values.
    tensor_shape_ = tf::TensorShape(std::vector<tf::int64>({1, 3, 1, 3, 1}));
    tensor_ = tf::Tensor(tf::DT_INT32, tensor_shape_);
    auto tensor_values = tensor_.tensor<int32, 5>();
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tensor_values(0, i, 0, j, 0) = i * (j + 1);
      }
    }
  }

  std::unique_ptr<CalculatorRunner> runner_;
  tf::TensorShape tensor_shape_;
  tf::Tensor tensor_;
};

TEST_F(TensorSqueezeDimensionsCalculatorTest, CanSqueezeAllSingleDimensions) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorSqueezeDimensionsCalculator");
  config.add_input_stream("input_tensor");
  config.add_output_stream("output_tensor");
  CalculatorOptions options;
  TensorSqueezeDimensionsCalculatorOptions* squeeze_options =
      options.MutableExtension(TensorSqueezeDimensionsCalculatorOptions::ext);
  squeeze_options->set_squeeze_all_single_dims(true);
  *config.mutable_options() = options;

  runner_.reset(new CalculatorRunner(config));
  std::unique_ptr<tf::Tensor> tensor_copy(new tf::Tensor);
  EXPECT_TRUE(tensor_copy->CopyFrom(tensor_, tensor_shape_));
  const tf::int64 time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor_copy.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();
  const tf::TensorShape expected_shape(std::vector<tf::int64>({3, 3}));
  EXPECT_EQ(expected_shape.DebugString(), output_tensor.shape().DebugString());
  const auto tensor_values = output_tensor.tensor<int32, 2>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int expected_value = i * (j + 1);
      EXPECT_EQ(expected_value, tensor_values(i, j));
    }
  }
}

TEST_F(TensorSqueezeDimensionsCalculatorTest, CanSqueezeSpecifiedDimensions) {
  CalculatorGraphConfig::Node config;
  config.set_calculator("TensorSqueezeDimensionsCalculator");
  config.add_input_stream("input_tensor");
  config.add_output_stream("output_tensor");
  CalculatorOptions options;
  TensorSqueezeDimensionsCalculatorOptions* squeeze_options =
      options.MutableExtension(TensorSqueezeDimensionsCalculatorOptions::ext);
  squeeze_options->add_dim(0);
  squeeze_options->add_dim(4);
  *config.mutable_options() = options;

  runner_.reset(new CalculatorRunner(config));
  std::unique_ptr<tf::Tensor> tensor_copy(new tf::Tensor);
  EXPECT_TRUE(tensor_copy->CopyFrom(tensor_, tensor_shape_));
  const tf::int64 time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor_copy.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const tf::Tensor& output_tensor = output_packets[0].Get<tf::Tensor>();
  const tf::TensorShape expected_shape(std::vector<tf::int64>({3, 1, 3}));
  EXPECT_EQ(expected_shape.DebugString(), output_tensor.shape().DebugString());
  const auto tensor_values = output_tensor.tensor<int32, 3>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      const int expected_value = i * (j + 1);
      EXPECT_EQ(expected_value, tensor_values(i, 0, j));
    }
  }
}

}  // namespace
}  // namespace mediapipe
