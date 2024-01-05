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

#include <cstdint>

#include "mediapipe/calculators/tensorflow/tensor_to_vector_string_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class TensorToVectorStringCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(const bool tensor_is_2d, const bool flatten_nd) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToVectorStringCalculator");
    config.add_input_stream("input_tensor");
    config.add_output_stream("output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        TensorToVectorStringCalculatorOptions::ext);
    options->set_tensor_is_2d(tensor_is_2d);
    options->set_flatten_nd(flatten_nd);
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorToVectorStringCalculatorTest, ConvertsToVectorFloat) {
  SetUpRunner(false, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_STRING, tensor_shape);
  auto tensor_vec = tensor->vec<tensorflow::tstring>();
  for (int i = 0; i < 5; ++i) {
    tensor_vec(i) = absl::StrCat("foo", i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<std::string>& output_vector =
      output_packets[0].Get<std::vector<std::string>>();

  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const std::string expected = absl::StrCat("foo", i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorStringCalculatorTest, ConvertsBatchedToVectorVectorFloat) {
  SetUpRunner(true, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{1, 5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_STRING, tensor_shape);
  auto slice = tensor->Slice(0, 1).flat<tensorflow::tstring>();
  for (int i = 0; i < 5; ++i) {
    slice(i) = absl::StrCat("foo", i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<std::vector<std::string>>& output_vectors =
      output_packets[0].Get<std::vector<std::vector<std::string>>>();
  ASSERT_EQ(1, output_vectors.size());
  const std::vector<std::string>& output_vector = output_vectors[0];
  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const std::string expected = absl::StrCat("foo", i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorStringCalculatorTest, FlattenShouldTakeAllDimensions) {
  SetUpRunner(false, true);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{2, 2, 2});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_STRING, tensor_shape);
  auto slice = tensor->flat<tensorflow::tstring>();
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    slice(i) = absl::StrCat("foo", i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<std::string>& output_vector =
      output_packets[0].Get<std::vector<std::string>>();
  EXPECT_EQ(2 * 2 * 2, output_vector.size());
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    const std::string expected = absl::StrCat("foo", i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

}  // namespace
}  // namespace mediapipe
