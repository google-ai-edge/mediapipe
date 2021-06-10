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

#include "absl/memory/memory.h"
#include "mediapipe/calculators/tensorflow/lapped_tensor_buffer_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace {

namespace tf = ::tensorflow;

class LappedTensorBufferCalculatorTest : public ::testing::Test {
 protected:
  void SetUpCalculator(int buffer_size, int overlap, bool add_dim,
                       int timestamp_offset, int padding,
                       bool timestamp_output) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("LappedTensorBufferCalculator");
    config.add_input_stream("input_tensor");
    config.add_output_stream("output_tensor");
    if (timestamp_output) {
      config.add_output_stream("output_timestamp");
    }
    auto options = config.mutable_options()->MutableExtension(
        LappedTensorBufferCalculatorOptions::ext);
    options->set_buffer_size(buffer_size);
    options->set_overlap(overlap);
    if (add_dim) {
      options->set_add_batch_dim_to_tensors(true);
    }
    options->set_timestamp_offset(timestamp_offset);
    options->set_padding(padding);
    runner_ = ::absl::make_unique<CalculatorRunner>(config);
  }
  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(LappedTensorBufferCalculatorTest, OneToOne) {
  SetUpCalculator(1, 0, false, 0, 0, false);
  int num_timesteps = 3;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps, output_packets.size());
  for (int i = 0; i < num_timesteps; ++i) {
    float value = output_packets[i].Get<tf::Tensor>().tensor<float, 1>()(0);
    ASSERT_NEAR(i, value, 0.0001);
  }
}

TEST_F(LappedTensorBufferCalculatorTest, OneToTwo) {
  int buffer_size = 2;
  int overlap = 1;
  bool add_dim = false;
  SetUpCalculator(buffer_size, overlap, add_dim, 0, 0, false);
  int num_timesteps = 3;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps - buffer_size + 1, output_packets.size());
  for (int i = 0; i < num_timesteps - buffer_size + 1; ++i) {
    for (int j = 0; j < buffer_size; ++j) {
      float value = output_packets[i].Get<tf::Tensor>().tensor<float, 1>()(j);
      ASSERT_NEAR(i + j, value, 0.0001);
    }
  }
}

TEST_F(LappedTensorBufferCalculatorTest, OneToThree) {
  int buffer_size = 3;
  int overlap = 2;
  bool add_dim = false;
  SetUpCalculator(buffer_size, overlap, add_dim, 0, 0, false);
  int num_timesteps = 3;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps - buffer_size + 1, output_packets.size());
  for (int i = 0; i < num_timesteps - buffer_size + 1; ++i) {
    for (int j = 0; j < buffer_size; ++j) {
      float value = output_packets[i].Get<tf::Tensor>().tensor<float, 1>()(j);
      ASSERT_NEAR(i + j, value, 0.0001);
    }
  }
}

TEST_F(LappedTensorBufferCalculatorTest, OneToThreeSkip) {
  int buffer_size = 3;
  int overlap = 1;
  bool add_dim = false;
  SetUpCalculator(buffer_size, overlap, add_dim, 0, 0, false);
  int num_timesteps = 3;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps - buffer_size + 1, output_packets.size());
  for (int i = 0; i < num_timesteps - buffer_size + 1; ++i) {
    for (int j = 0; j < buffer_size; ++j) {
      float value = output_packets[i].Get<tf::Tensor>().tensor<float, 1>()(j);
      ASSERT_NEAR((i * 2) + j, value, 0.0001);
    }
  }
}

TEST_F(LappedTensorBufferCalculatorTest, OneToThreeNegativeOverlap) {
  int buffer_size = 3;
  int overlap = -1;
  bool add_dim = false;
  SetUpCalculator(buffer_size, overlap, add_dim, 0, 0, false);
  int num_timesteps = 7;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(2, output_packets.size());
  // The outputs in packet one should be {0, 1, 2}, and in packet two {4, 5, 6}
  for (int i = 0; i < 3; ++i) {
    float value_0 = output_packets[0].Get<tf::Tensor>().tensor<float, 1>()(i);
    ASSERT_NEAR(value_0, i, 0.0001);
  }
  for (int i = 0; i < 3; ++i) {
    float value_1 = output_packets[1].Get<tf::Tensor>().tensor<float, 1>()(i);
    ASSERT_NEAR(value_1, 4 + i, 0.0001);
  }
}

TEST_F(LappedTensorBufferCalculatorTest, OneToThreeBatch) {
  int buffer_size = 3;
  int overlap = 2;
  bool add_dim = true;
  SetUpCalculator(buffer_size, overlap, add_dim, 0, 0, false);
  int num_timesteps = 3;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps - buffer_size + 1, output_packets.size());
  for (int i = 0; i < num_timesteps - buffer_size + 1; ++i) {
    for (int j = 0; j < buffer_size; ++j) {
      float value =
          output_packets[i].Get<tf::Tensor>().tensor<float, 2>()(j, 0);
      ASSERT_NEAR(i + j, value, 0.0001);
    }
  }
}

TEST_F(LappedTensorBufferCalculatorTest, NegativeTimestampOffsetFails) {
  int buffer_size = 16;
  int overlap = 15;
  bool add_dim = true;
  int timestamp_offset = -7;
  SetUpCalculator(buffer_size, overlap, add_dim, timestamp_offset, 0, false);
  int num_timesteps = 20;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_FALSE(runner_->Run().ok());
}

TEST_F(LappedTensorBufferCalculatorTest, OutOfRangeTimestampOffsetFails) {
  int buffer_size = 16;
  int overlap = 15;
  bool add_dim = true;
  int timestamp_offset = buffer_size;
  SetUpCalculator(buffer_size, overlap, add_dim, timestamp_offset, 0, false);
  int num_timesteps = 20;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_FALSE(runner_->Run().ok());
}

TEST_F(LappedTensorBufferCalculatorTest, OneToThreeBatchTimestampOffset) {
  int buffer_size = 16;
  int overlap = 15;
  bool add_dim = true;
  int timestamp_offset = 7;
  SetUpCalculator(buffer_size, overlap, add_dim, timestamp_offset, 0, false);
  int num_timesteps = 20;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(num_timesteps - buffer_size + 1, output_packets.size());
  for (int i = 0; i < num_timesteps - buffer_size + 1; ++i) {
    for (int j = 0; j < buffer_size; ++j) {
      int64 value = output_packets[i].Timestamp().Value();
      ASSERT_EQ(i + timestamp_offset, value);
    }
  }
}

TEST_F(LappedTensorBufferCalculatorTest,
       OneToThreeBatchTimestampOffsetPadding) {
  int buffer_size = 12;
  int overlap = 6;
  bool add_dim = true;
  int timestamp_offset = 3;
  int padding = 0;
  SetUpCalculator(buffer_size, overlap, add_dim, timestamp_offset, padding,
                  true);
  int num_timesteps = 20;
  for (int i = 0; i < num_timesteps; ++i) {
    auto input = ::absl::make_unique<tensorflow::Tensor>(
        tensorflow::DT_FLOAT, tensorflow::TensorShape({1}));
    input->tensor<float, 1>()(0) = i;
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(input.release()).At(Timestamp(i)));
  }
  ASSERT_TRUE(runner_->Run().ok());

  const int output_size = num_timesteps / buffer_size + 1;
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(output_size, output_packets.size());
  for (int i = 0; i < output_size; ++i) {
    int64 value = output_packets[i].Timestamp().Value();
    ASSERT_EQ(i * overlap + timestamp_offset, value);
  }
  const std::vector<Packet>& output_timestamps =
      runner_->Outputs().Index(1).packets;
  ASSERT_EQ(output_size, output_timestamps.size());
}

}  // namespace
}  // namespace mediapipe
