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

#include "mediapipe/calculators/tensorflow/tensor_to_vector_int_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class TensorToVectorIntCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(const bool tensor_is_2d, const bool flatten_nd,
                   const bool tensor_is_token = false,
                   const int32_t overlap = 0) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToVectorIntCalculator");
    config.add_input_stream("input_tensor");
    config.add_output_stream("output_tensor");
    auto options = config.mutable_options()->MutableExtension(
        TensorToVectorIntCalculatorOptions::ext);
    options->set_tensor_is_2d(tensor_is_2d);
    options->set_flatten_nd(flatten_nd);
    options->set_tensor_is_token(tensor_is_token);
    options->set_overlap(overlap);
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorToVectorIntCalculatorTest, ConvertsToVectorInt) {
  SetUpRunner(false, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT64, tensor_shape);
  auto tensor_vec = tensor->vec<int64_t>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<int64_t>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  ASSERT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<int64_t>& output_vector =
      output_packets[0].Get<std::vector<int64_t>>();

  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const int64_t expected = static_cast<int64_t>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorIntCalculatorTest, ConvertsToVectorFromInt32) {
  SetUpRunner(false, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT32, tensor_shape);
  auto tensor_vec = tensor->vec<int32_t>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<int32_t>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  ASSERT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<int64_t>& output_vector =
      output_packets[0].Get<std::vector<int64_t>>();

  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const int64_t expected = static_cast<int64_t>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorIntCalculatorTest, ConvertsToVectorToken) {
  SetUpRunner(false, false, true);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT32, tensor_shape);
  auto tensor_vec = tensor->vec<int32_t>();
  tensor_vec(0) = 0;
  tensor_vec(1) = 0;
  tensor_vec(2) = 1;
  tensor_vec(3) = 1;
  tensor_vec(4) = 0;

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  ASSERT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<int64_t>& output_vector =
      output_packets[0].Get<std::vector<int64_t>>();

  EXPECT_EQ(2, output_vector.size());
  EXPECT_EQ(3, output_vector[0]);
  EXPECT_EQ(4, output_vector[1]);
}

TEST_F(TensorToVectorIntCalculatorTest, ConvertsBatchedToVectorVectorInt) {
  SetUpRunner(true, false);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{1, 5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT64, tensor_shape);
  auto slice = tensor->Slice(0, 1).flat<int64_t>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    slice(i) = static_cast<int64_t>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<std::vector<int64_t>>& output_vectors =
      output_packets[0].Get<std::vector<std::vector<int64_t>>>();
  ASSERT_EQ(1, output_vectors.size());
  const std::vector<int64_t>& output_vector = output_vectors[0];
  EXPECT_EQ(5, output_vector.size());
  for (int i = 0; i < 5; ++i) {
    const int64_t expected = static_cast<int64_t>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorIntCalculatorTest, FlattenShouldTakeAllDimensions) {
  SetUpRunner(false, true);
  const tf::TensorShape tensor_shape(std::vector<int64_t>{2, 2, 2});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT64, tensor_shape);
  auto slice = tensor->flat<int64_t>();
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    slice(i) = static_cast<int64_t>(1 << i);
  }

  const int64_t time = 1234;
  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const std::vector<int64_t>& output_vector =
      output_packets[0].Get<std::vector<int64_t>>();
  EXPECT_EQ(2 * 2 * 2, output_vector.size());
  for (int i = 0; i < 2 * 2 * 2; ++i) {
    const int64_t expected = static_cast<int64_t>(1 << i);
    EXPECT_EQ(expected, output_vector[i]);
  }
}

TEST_F(TensorToVectorIntCalculatorTest, Overlap) {
  SetUpRunner(false, false, false, 2);
  for (int time = 0; time < 3; ++time) {
    const tf::TensorShape tensor_shape(std::vector<int64_t>{5});
    auto tensor = absl::make_unique<tf::Tensor>(tf::DT_INT64, tensor_shape);
    auto tensor_vec = tensor->vec<int64_t>();
    for (int i = 0; i < 5; ++i) {
      // 2^i can be represented exactly in floating point numbers if 'i' is
      // small.
      tensor_vec(i) = static_cast<int64_t>(time + (1 << i));
    }

    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(tensor.release()).At(Timestamp(time)));
  }

  ASSERT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  EXPECT_EQ(3, output_packets.size());

  {
    // First vector in full.
    int time = 0;
    EXPECT_EQ(time, output_packets[time].Timestamp().Value());
    const std::vector<int64_t>& output_vector =
        output_packets[time].Get<std::vector<int64_t>>();

    EXPECT_EQ(5, output_vector.size());
    for (int i = 0; i < 5; ++i) {
      const int64_t expected = static_cast<int64_t>(time + (1 << i));
      EXPECT_EQ(expected, output_vector[i]);
    }
  }

  // All following vectors the overlap removed
  for (int time = 1; time < 3; ++time) {
    EXPECT_EQ(time, output_packets[time].Timestamp().Value());
    const std::vector<int64_t>& output_vector =
        output_packets[time].Get<std::vector<int64_t>>();

    EXPECT_EQ(3, output_vector.size());
    for (int i = 0; i < 3; ++i) {
      const int64_t expected = static_cast<int64_t>(time + (1 << (i + 2)));
      EXPECT_EQ(expected, output_vector[i]);
    }
  }
}

}  // namespace
}  // namespace mediapipe
