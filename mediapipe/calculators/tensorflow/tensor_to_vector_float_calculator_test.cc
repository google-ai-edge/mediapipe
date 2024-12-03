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
#include <memory>

#include "mediapipe/calculators/tensorflow/tensor_to_vector_float_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/util/packet_test_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/bfloat16.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class TensorToVectorFloatCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(const bool tensor_is_2d, const bool flatten_nd) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToVectorFloatCalculator");
    config.add_input_stream("input_tensor");
    config.add_output_stream("output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        TensorToVectorFloatCalculatorOptions::ext);
    options->set_tensor_is_2d(tensor_is_2d);
    options->set_flatten_nd(flatten_nd);
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorToVectorFloatCalculatorTest, ConvertsToVectorFloat) {
  SetUpRunner(false, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->vec<float>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<float>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<float>& output_vector =
      output_packets[0].Get<std::vector<float>>();

  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorFloatCalculatorTest, CheckBFloat16Type) {
  SetUpRunner(false, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = std::make_unique<tf::Tensor>(tf::DT_BFLOAT16, tensor_shape);
  auto tensor_vec = tensor->vec<tf::bfloat16>();
  for (int i = 0; i < 5; ++i) {
    tensor_vec(i) = static_cast<tf::bfloat16>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<float>& output_vector =
      output_packets[0].Get<std::vector<float>>();

  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorFloatCalculatorTest, CheckBFloat16TypeAllDim) {
  SetUpRunner(false, true);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{2, 2, 2});
  auto tensor = std::make_unique<tf::Tensor>(tf::DT_BFLOAT16, tensor_shape);
  auto slice = tensor->flat<tf::bfloat16>();
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    slice(i) = static_cast<tf::bfloat16>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<float>& output_vector =
      output_packets[0].Get<std::vector<float>>();
  EXPECT_EQ(2 * 2 * 2, output_vector.size());
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorFloatCalculatorTest, ConvertsBatchedToVectorVectorFloat) {
  SetUpRunner(true, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{1, 5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto slice = tensor->Slice(0, 1).flat<float>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    slice(i) = static_cast<float>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<std::vector<float>>& output_vectors =
      output_packets[0].Get<std::vector<std::vector<float>>>();
  ASSERT_EQ(1, output_vectors.size());
  const std::vector<float>& output_vector = output_vectors[0];
  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorFloatCalculatorTest, FlattenShouldTakeAllDimensions) {
  SetUpRunner(false, true);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{2, 2, 2});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto slice = tensor->flat<float>();
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    slice(i) = static_cast<float>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<float>& output_vector =
      output_packets[0].Get<std::vector<float>>();
  EXPECT_EQ(2 * 2 * 2, output_vector.size());
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorFloatCalculatorTest, AcceptsUnalignedTensors) {
  SetUpRunner(/*tensor_is_2d=*/false, /*flatten_nd=*/false);

  const tf::TensorShape tensor_shape(std::vector<int64_t>{2, 5});
  tf::Tensor tensor(tf::DT_FLOAT, tensor_shape);
  auto slice = tensor.Slice(1, 1).flat<float>();
  for (int i = 0; i < 5; ++i) {
    slice(i) = i;
  }

  auto input_tensor = tensor.SubSlice(1);
  // Ensure that the input tensor is unaligned.
  ASSERT_FALSE(input_tensor.IsAligned());
  runner_->MutableInputs()->Index(0).packets.push_back(
      MakePacket<tf::Tensor>(input_tensor).At(Timestamp(5)));

  ASSERT_TRUE(runner_->Run().ok());

  EXPECT_THAT(runner_->Outputs().Index(0).packets,
              ElementsAre(PacketContainsTimestampAndPayload<std::vector<float>>(
                  Timestamp(5), std::vector<float>({0, 1, 2, 3, 4}))));
}

}  // namespace
}  // namespace mediapipe
