// Copyright 2019 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensorflow/matrix_to_tensor_calculator_options.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace {

constexpr char kTransposeOptionsString[] =
    "[mediapipe.MatrixToTensorCalculatorOptions.ext]: {"
    "transpose: True}";
constexpr char kAddDimensionOptionsString[] =
    "[mediapipe.MatrixToTensorCalculatorOptions.ext]: {"
    "add_trailing_dimension: True}";

}  // namespace

namespace tf = tensorflow;
using RandomEngine = std::mt19937_64;
const uint32 kSeed = 1234;
const int kNumSizes = 8;
const int sizes[kNumSizes][2] = {{1, 1}, {12, 1}, {1, 9},   {2, 2},
                                 {5, 3}, {7, 13}, {16, 32}, {101, 2}};

class MatrixToTensorCalculatorTest : public ::testing::Test {
 protected:
  // Adds a packet with a matrix filled with random values in [0,1].
  void AddRandomMatrix(int num_rows, int num_columns, uint32 seed) {
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    auto matrix = ::absl::make_unique<Matrix>();
    matrix->resize(num_rows, num_columns);
    for (int y = 0; y < num_rows; ++y) {
      for (int x = 0; x < num_columns; ++x) {
        (*matrix)(y, x) = uniform_dist(random);
      }
    }
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(matrix.release()).At(Timestamp(0)));
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(MatrixToTensorCalculatorTest, RandomMatrix) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    runner_ = ::absl::make_unique<CalculatorRunner>("MatrixToTensorCalculator",
                                                    "", 1, 1, 0);
    AddRandomMatrix(num_rows, num_columns, kSeed);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that the packet contains a 2D float tensor.
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(2, tensor.dims());
    ASSERT_EQ(tf::DT_FLOAT, tensor.dtype());

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const auto matrix = tensor.matrix<float>();
    for (int y = 0; y < num_rows; ++y) {
      for (int x = 0; x < num_columns; ++x) {
        const float expected = uniform_dist(random);
        ASSERT_EQ(expected, matrix(y, x));
      }
    }
  }
}

TEST_F(MatrixToTensorCalculatorTest, RandomMatrixTranspose) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    runner_ = ::absl::make_unique<CalculatorRunner>(
        "MatrixToTensorCalculator", kTransposeOptionsString, 1, 1, 0);
    AddRandomMatrix(num_rows, num_columns, kSeed);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that the packet contains a 2D float tensor.
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(2, tensor.dims());
    ASSERT_EQ(tf::DT_FLOAT, tensor.dtype());

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    const auto matrix = tensor.matrix<float>();
    for (int y = 0; y < num_rows; ++y) {
      for (int x = 0; x < num_columns; ++x) {
        const float expected = uniform_dist(random);
        ASSERT_EQ(expected, matrix(x, y));
      }
    }
  }
}

TEST_F(MatrixToTensorCalculatorTest, RandomMatrixAddDimension) {
  for (int size_index = 0; size_index < kNumSizes; ++size_index) {
    const int num_rows = sizes[size_index][0];
    const int num_columns = sizes[size_index][1];

    // Run the calculator and verify that one output is generated.
    runner_ = ::absl::make_unique<CalculatorRunner>(
        "MatrixToTensorCalculator", kAddDimensionOptionsString, 1, 1, 0);
    AddRandomMatrix(num_rows, num_columns, kSeed);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that the packet contains a 3D float tensor.
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_FLOAT, tensor.dtype());

    // Verify that the data is correct.
    RandomEngine random(kSeed);
    std::uniform_real_distribution<> uniform_dist(0, 1.0);
    // const auto matrix = tensor.matrix<float>();
    const float* tensor_data = tensor.flat<float>().data();
    for (int y = 0; y < num_rows; ++y) {
      for (int x = 0; x < num_columns; ++x) {
        const float expected = uniform_dist(random);
        ASSERT_EQ(expected, tensor_data[y * num_columns + x]);
      }
    }
  }
}

}  // namespace mediapipe
