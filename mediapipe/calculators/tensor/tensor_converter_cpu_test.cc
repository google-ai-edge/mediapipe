// Copyright 2023 The MediaPipe Authors.
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

#include "mediapipe/calculators/tensor/tensor_converter_cpu.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "mediapipe/framework/formats/matrix.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

Matrix CreateTestMatrix(int num_rows, int num_columns) {
  Matrix matrix(num_rows, num_columns);
  for (int r = 0; r < num_rows; ++r) {
    for (int c = 0; c < num_columns; ++c) {
      matrix(r, c) = r * num_columns + c;
    }
  }
  return matrix;
}

TEST(TensorConverterCpuTest, ShouldCopyMatrixInRowMajorFormatToTensor) {
  auto test_matrix = CreateTestMatrix(/* num_rows=*/3, /*num_columns=*/4);
  std::vector<float> tensor_data(test_matrix.size(), 0.0f);

  MP_EXPECT_OK(CopyMatrixToTensor(test_matrix, /*is_row_major_matrix=*/true,
                                  tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    const int row = i / test_matrix.cols();
    const int column = i % test_matrix.cols();
    EXPECT_FLOAT_EQ(tensor_data[i], (test_matrix)(row, column));
  }
}

TEST(TensorConverterCpuTest, ShouldCopyMatrixInColumnMajorFormatToTensor) {
  auto test_matrix = CreateTestMatrix(/*num_rows=*/3, /*num_columns=*/4);
  std::vector<float> tensor_data(test_matrix.size(), 0.0f);

  MP_EXPECT_OK(CopyMatrixToTensor(test_matrix, /*is_row_major_matrix=*/false,
                                  tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    const int row = i % test_matrix.rows();
    const int column = i / test_matrix.rows();
    EXPECT_FLOAT_EQ(tensor_data[i], (test_matrix)(row, column));
  }
}

TEST(TensorConverterCpuTest, ShouldNormalizeGrey8ImageWithDefaultRange) {
  auto grey8_image_frame = CreateTestGrey8ImageFrame(/*width=*/3, /*height=*/4);
  std::vector<float> tensor_data(
      grey8_image_frame.Width() * grey8_image_frame.Height(), 0.0f);

  MP_EXPECT_OK(NormalizeUInt8Image(grey8_image_frame, /*flip_vertically=*/false,
                                   {0.0f, 1.0f}, /*num_tensor_channels=*/1,
                                   tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    EXPECT_FLOAT_EQ(
        tensor_data[i],
        static_cast<uint8_t>(grey8_image_frame.PixelData()[i]) / 255.0f);
  }
}

TEST(TensorConverterCpuTest, ShouldNormalizeGrey8ImageWithSpecifiedRange) {
  auto grey8_image_frame = CreateTestGrey8ImageFrame(/*width=*/3, /*height=*/4);
  std::vector<float> tensor_data(
      grey8_image_frame.Width() * grey8_image_frame.Height(), 0.0f);
  const auto range = std::make_pair(2.0f, 3.0f);

  MP_EXPECT_OK(
      NormalizeUInt8Image(grey8_image_frame, /*flip_vertically=*/false, range,
                          /*num_tensor_channels=*/1, tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i],
                    static_cast<uint8_t>(grey8_image_frame.PixelData()[i]) /
                            255.0f * (range.second - range.first) +
                        range.first);
  }
}

TEST(TensorConverterCpuTest, ShouldNormalizeGrey8ImageFlipped) {
  auto grey8_image_frame = CreateTestGrey8ImageFrame(/*width=*/3, /*height=*/4);
  std::vector<float> tensor_data(
      grey8_image_frame.Width() * grey8_image_frame.Height(), 0.0f);

  MP_EXPECT_OK(NormalizeUInt8Image(grey8_image_frame, /*flip_vertically=*/true,
                                   {0.0f, 1.0f}, /*num_tensor_channels=*/1,
                                   tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    const int x = i % grey8_image_frame.Width();
    const int y = i / grey8_image_frame.Width();
    const int flipped_y = grey8_image_frame.Height() - y - 1;

    const int index = flipped_y * grey8_image_frame.Width() + x;
    EXPECT_FLOAT_EQ(
        tensor_data[index],
        static_cast<uint8_t>(grey8_image_frame.PixelData()[i]) / 255.0f);
  }
}

TEST(TensorConverterCpuTest, ShouldNormalizeFloatImageWithDefaultRange) {
  auto float_image_frame =
      CreateTestFloat32ImageFrame(/*width=*/3, /*height=*/4);
  std::vector<float> tensor_data(
      float_image_frame.Width() * float_image_frame.Height(), 0.0f);

  MP_EXPECT_OK(NormalizeFloatImage(float_image_frame, /*flip_vertically=*/false,
                                   {0.0f, 1.0f}, /*num_tensor_channels=*/1,
                                   tensor_data.data()));

  for (int i = 0; i < tensor_data.size(); ++i) {
    EXPECT_FLOAT_EQ(tensor_data[i], reinterpret_cast<const float*>(
                                        float_image_frame.PixelData())[i] /
                                        255.0f);
  }
}

TEST(TensorConverterCpuTest, ConvertImageFrameToTensorOnCpu) {
  MemoryManager memory_manager;
  auto grey8_image_frame = CreateTestGrey8ImageFrame(/*width=*/3, /*height=*/4);

  MP_ASSERT_OK_AND_ASSIGN(
      Tensor output,
      ConvertImageFrameToTensorOnCpu(grey8_image_frame, {0.0f, 1.0f},
                                     /*flip_vertically=*/false,
                                     /*max_num_channels=*/1, &memory_manager));

  const auto cpu_read_view = output.GetCpuReadView();
  const float* tensor_ptr = cpu_read_view.buffer<float>();
  for (int i = 0; i < grey8_image_frame.Width() * grey8_image_frame.Height();
       ++i) {
    EXPECT_FLOAT_EQ(
        tensor_ptr[i],
        static_cast<uint8_t>(grey8_image_frame.PixelData()[i]) / 255.0);
  }
}

TEST(TensorConverterCpuTest, ConvertMatrixToTensorOnCpu) {
  MemoryManager memory_manager;
  auto test_matrix = CreateTestMatrix(/*num_rows=*/3, /*num_columns=*/4);

  MP_ASSERT_OK_AND_ASSIGN(
      Tensor output,
      ConvertMatrixToTensorOnCpu(test_matrix,
                                 /*row_major_matrix=*/false, &memory_manager));

  const auto cpu_read_view = output.GetCpuReadView();
  const float* tensor_ptr = cpu_read_view.buffer<float>();
  for (int i = 0; i < test_matrix.size(); ++i) {
    EXPECT_FLOAT_EQ(tensor_ptr[i], test_matrix.data()[i]);
  }
}

}  // namespace

}  // namespace mediapipe
