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

#include "mediapipe/calculators/tensorflow/tensor_to_matrix_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/time_series_header.pb.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

constexpr char kReferenceTag[] = "REFERENCE";

constexpr char kMatrix[] = "MATRIX";
constexpr char kTensor[] = "TENSOR";

}  // namespace

class TensorToMatrixCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner() {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToMatrixCalculator");
    config.add_input_stream("TENSOR:input_tensor");
    config.add_output_stream("MATRIX:output_matrix");
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  // Creates a reference stream and sets num_channels and num_samples if > 0.
  void SetUpRunnerWithReference(int channels, int samples,
                                int override_channels, bool include_rate) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToMatrixCalculator");
    config.add_input_stream("TENSOR:input_tensor");
    config.add_input_stream("REFERENCE:reference");
    config.add_output_stream("MATRIX:output_matrix");
    if (override_channels > 0) {
      config.mutable_options()
          ->MutableExtension(TensorToMatrixCalculatorOptions::ext)
          ->mutable_time_series_header_overrides()
          ->set_num_channels(override_channels);
    }
    runner_ = absl::make_unique<CalculatorRunner>(config);

    auto header = absl::make_unique<TimeSeriesHeader>();
    header->set_sample_rate(1.0);
    if (channels > 0) {
      header->set_num_channels(channels);
    }
    if (samples > 0) {
      header->set_num_samples(samples);
    }
    if (include_rate) {
      header->set_packet_rate(1.0);
    }
    runner_->MutableInputs()->Tag(kReferenceTag).header =
        Adopt(header.release());
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorToMatrixCalculatorTest, Converts1DTensorToMatrix) {
  // This test converts a 1 Dimensional Tensor of length M to a Matrix of Mx1.
  SetUpRunner();
  const tf::TensorShape tensor_shape(std::vector<tf::int64>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->vec<float>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<float>(1 << i);
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kMatrix).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const Matrix& output_matrix = output_packets[0].Get<Matrix>();
  EXPECT_EQ(5, output_matrix.rows());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_matrix(i, 0));
  }
}

TEST_F(TensorToMatrixCalculatorTest, Converts2DTensorofWidthOneToMatrix) {
  // This test converts a 2 Dimensional Tensor of shape 1xM to a Matrix of Mx1.
  SetUpRunner();
  const tf::TensorShape tensor_shape(std::vector<tf::int64>({1, 4}));
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto slice = tensor->Slice(0, 1).flat<float>();
  for (int i = 0; i < 4; ++i) {
    slice(i) = static_cast<float>(1 << i);
  }
  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kMatrix).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const Matrix& output_matrix = output_packets[0].Get<Matrix>();
  ASSERT_EQ(1, output_matrix.cols());
  EXPECT_EQ(4, output_matrix.rows());
  for (int i = 0; i < 4; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_matrix(i, 0));
  }
}

TEST_F(TensorToMatrixCalculatorTest, Converts2DTensorToMatrix) {
  // This test converts a 2 Dimensional Tensor of shape NxM to a Matrix of MxN.
  SetUpRunner();
  const tf::TensorShape tensor_shape(std::vector<tf::int64>({3, 4}));
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto slice = tensor->Slice(0, 1).flat<float>();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      slice(i * 4 + j) = static_cast<float>(i * j);
    }
  }
  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kMatrix).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const Matrix& output_matrix = output_packets[0].Get<Matrix>();
  ASSERT_EQ(3, output_matrix.cols());
  EXPECT_EQ(4, output_matrix.rows());
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      const float expected = static_cast<float>(i * j);
      EXPECT_EQ(expected, output_matrix(i, j));
    }
  }
}

TEST_F(TensorToMatrixCalculatorTest, ConvertsWithReferenceTimeSeriesHeader) {
  // This test converts a 1 Dimensional Tensor of length M to a Matrix of Mx1.
  SetUpRunnerWithReference(5, 1, -1, true);
  const tf::TensorShape tensor_shape(std::vector<tf::int64>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->vec<float>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<float>(1 << i);
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kMatrix).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const Matrix& output_matrix = output_packets[0].Get<Matrix>();
  EXPECT_EQ(5, output_matrix.rows());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_matrix(i, 0));
  }

  const TimeSeriesHeader& output_header =
      runner_->Outputs().Tag(kMatrix).header.Get<TimeSeriesHeader>();
  EXPECT_EQ(output_header.num_channels(), 5);
}

TEST_F(TensorToMatrixCalculatorTest, TimeSeriesOverridesWork) {
  // This test converts a 1 Dimensional Tensor of length M to a Matrix of Mx1.
  SetUpRunnerWithReference(7, 1, 5, true);
  const tf::TensorShape tensor_shape(std::vector<tf::int64>{5});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->vec<float>();
  for (int i = 0; i < 5; ++i) {
    // 2^i can be represented exactly in floating point numbers if 'i' is small.
    tensor_vec(i) = static_cast<float>(1 << i);
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kMatrix).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const Matrix& output_matrix = output_packets[0].Get<Matrix>();
  EXPECT_EQ(5, output_matrix.rows());
  for (int i = 0; i < 5; ++i) {
    const float expected = static_cast<float>(1 << i);
    EXPECT_EQ(expected, output_matrix(i, 0));
  }

  const TimeSeriesHeader& output_header =
      runner_->Outputs().Tag(kMatrix).header.Get<TimeSeriesHeader>();
  EXPECT_EQ(output_header.num_channels(), 5);
}

}  // namespace mediapipe
