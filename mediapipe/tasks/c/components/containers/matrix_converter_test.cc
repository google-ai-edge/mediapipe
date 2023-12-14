/* Copyright 2023 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "mediapipe/tasks/c/components/containers/matrix_converter.h"

#include "Eigen/Core"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/tasks/c/components/containers/matrix.h"

namespace mediapipe::tasks::c::components::containers {

TEST(MatrixConversionTest, ConvertsEigenMatrixToCMatrixAndFreesMemory) {
  // Initialize an Eigen::MatrixXf
  Eigen::MatrixXf cpp_matrix(2, 2);
  cpp_matrix << 1.0f, 2.0f, 3.0f, 4.0f;

  // Convert this Eigen matrix to C-style Matrix
  ::Matrix c_matrix;
  CppConvertToMatrix(cpp_matrix, &c_matrix);

  // Verify the conversion
  EXPECT_EQ(c_matrix.rows, 2);
  EXPECT_EQ(c_matrix.cols, 2);
  ASSERT_NE(c_matrix.data, nullptr);
  EXPECT_FLOAT_EQ(c_matrix.data[0], 1.0f);
  EXPECT_FLOAT_EQ(c_matrix.data[1], 3.0f);
  EXPECT_FLOAT_EQ(c_matrix.data[2], 2.0f);
  EXPECT_FLOAT_EQ(c_matrix.data[3], 4.0f);

  // Close the C-style Matrix
  CppCloseMatrix(&c_matrix);

  // Verify that memory is freed
  EXPECT_EQ(c_matrix.data, nullptr);
}

}  // namespace mediapipe::tasks::c::components::containers
