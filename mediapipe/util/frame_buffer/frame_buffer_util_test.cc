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

#include "mediapipe/util/frame_buffer/frame_buffer_util.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/log/absl_check.h"
#include "mediapipe/framework/formats/frame_buffer.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/framework/port/status_matchers.h"

namespace mediapipe {
namespace frame_buffer {
namespace {

// Grayscale unit tests.
//------------------------------------------------------------------------------

TEST(FrameBufferUtil, GrayCrop) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kOutputDimension = {.width = 1, .height = 1};
  uint8_t data[6] = {1, 2, 3, 4, 5, 6};
  uint8_t output_data[2];
  auto input = CreateFromGrayRawBuffer(data, kBufferDimension);
  auto output = CreateFromGrayRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Crop(*input, 0, 1, 0, 1, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 4);
}

TEST(FrameBufferUtil, GrayResize) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 2, .height = 2},
                                   kOutputDimension = {.width = 3, .height = 2};
  uint8_t data[4] = {1, 2, 3, 4};
  uint8_t output_data[6];
  auto input = CreateFromGrayRawBuffer(data, kBufferDimension);
  auto output = CreateFromGrayRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Resize(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 1);
  EXPECT_EQ(output->plane(0).buffer()[1], 2);
  EXPECT_EQ(output->plane(0).buffer()[2], 2);
  EXPECT_EQ(output->plane(0).buffer()[3], 3);
  EXPECT_EQ(output->plane(0).buffer()[4], 4);
  EXPECT_EQ(output->plane(0).buffer()[5], 4);
}

TEST(FrameBufferUtil, GrayRotate) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kOutputDimension = {.width = 2, .height = 3};
  uint8_t data[6] = {1, 2, 3, 4, 5, 6};
  uint8_t output_data[6];
  auto input = CreateFromGrayRawBuffer(data, kBufferDimension);
  auto output = CreateFromGrayRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Rotate(*input, 90, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 3);
  EXPECT_EQ(output->plane(0).buffer()[1], 6);
  EXPECT_EQ(output->plane(0).buffer()[2], 2);
  EXPECT_EQ(output->plane(0).buffer()[3], 5);
  EXPECT_EQ(output->plane(0).buffer()[4], 1);
  EXPECT_EQ(output->plane(0).buffer()[5], 4);
}

TEST(FrameBufferUtil, GrayFlipHorizontally) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2};
  uint8_t data[6] = {1, 2, 3, 4, 5, 6};
  uint8_t output_data[6];
  auto input = CreateFromGrayRawBuffer(data, kBufferDimension);
  auto output = CreateFromGrayRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipHorizontally(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 3);
  EXPECT_EQ(output->plane(0).buffer()[1], 2);
  EXPECT_EQ(output->plane(0).buffer()[2], 1);
  EXPECT_EQ(output->plane(0).buffer()[3], 6);
  EXPECT_EQ(output->plane(0).buffer()[4], 5);
  EXPECT_EQ(output->plane(0).buffer()[5], 4);
}

TEST(FrameBufferUtil, GrayFlipVertically) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2};
  uint8_t data[6] = {1, 2, 3, 4, 5, 6};
  uint8_t output_data[6];
  auto input = CreateFromGrayRawBuffer(data, kBufferDimension);
  auto output = CreateFromGrayRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipVertically(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 4);
  EXPECT_EQ(output->plane(0).buffer()[1], 5);
  EXPECT_EQ(output->plane(0).buffer()[2], 6);
  EXPECT_EQ(output->plane(0).buffer()[3], 1);
  EXPECT_EQ(output->plane(0).buffer()[4], 2);
  EXPECT_EQ(output->plane(0).buffer()[5], 3);
}

// Grayscale EndToEnd tests.
//------------------------------------------------------------------------------

struct GrayInputTestParam {
  FrameBuffer::Dimension input_dimension;
  FrameBuffer::Format input_format;
  FrameBuffer::Dimension output_dimension;
  FrameBuffer::Format output_format;
  int rotation_angle;
  int x0;
  int y0;
  int x1;
  int y1;
};

enum Operation {
  kRotate = 1,
  kCrop = 2,
  kResize = 3,
  kHorizontalFlip = 4,
  kVerticalFlip = 5,
  kConvert = 6
};

class GrayInputTest : public ::testing::TestWithParam<
                          std::tuple<Operation, GrayInputTestParam, bool>> {};

TEST_P(GrayInputTest, ValidateInputs) {
  GrayInputTestParam inputs;
  bool is_valid;
  Operation operation;
  std::tie(operation, inputs, is_valid) = GetParam();
  MP_ASSERT_OK_AND_ASSIGN(
      auto input,
      CreateFromRawBuffer(/*buffer=*/nullptr, inputs.input_dimension,
                          inputs.input_format));
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(nullptr, inputs.output_dimension,
                                              inputs.output_format));

  absl::Status status;
  switch (operation) {
    case kRotate:
      status = Rotate(*input, inputs.rotation_angle, output.get());
      break;
    case kResize:
      status = Resize(*input, output.get());
      break;
    case kCrop: {
      status = Crop(*input, inputs.x0, inputs.y0, inputs.x1, inputs.y1,
                    output.get());
      break;
    }
    case kHorizontalFlip:
      status = FlipHorizontally(*input, output.get());
      break;
    case kVerticalFlip:
      status = FlipVertically(*input, output.get());
      break;
    case kConvert:
      status = Convert(*input, output.get());
      break;
  }

  if (is_valid) {
    MP_EXPECT_OK(status);
  } else {
    EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

std::tuple<Operation, GrayInputTestParam, bool> CreateGrayRotateInputTestParam(
    int in_width, int in_height, FrameBuffer::Format in_format, int out_width,
    int out_height, FrameBuffer::Format out_format, int angle, bool is_valid) {
  GrayInputTestParam param = {
      .input_dimension = FrameBuffer::Dimension{in_width, in_height},
      .input_format = in_format,
      .output_dimension = FrameBuffer::Dimension{out_width, out_height},
      .output_format = out_format,
      .rotation_angle = angle};
  return std::make_tuple(kRotate, param, is_valid);
}

INSTANTIATE_TEST_SUITE_P(
    ValidateRotateInputs, GrayInputTest,
    testing::Values(
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 3,
                                       FrameBuffer::Format::kGRAY, 30, false),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                       FrameBuffer::Format::kRGB, 180, false),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                       FrameBuffer::Format::kGRAY, 90, false),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                       FrameBuffer::Format::kGRAY, 0, false),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 3,
                                       FrameBuffer::Format::kGRAY, -90, false),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 3,
                                       FrameBuffer::Format::kGRAY, 90, true),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                       FrameBuffer::Format::kGRAY, 180, true),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 3,
                                       FrameBuffer::Format::kGRAY, 270, true),
        CreateGrayRotateInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 3,
                                       FrameBuffer::Format::kGRAY, 450,
                                       false)));

std::tuple<Operation, GrayInputTestParam, bool> CreateGrayCropInputTestParam(
    int in_width, int in_height, FrameBuffer::Format in_format, int out_width,
    int out_height, FrameBuffer::Format out_format, int x0, int y0, int x1,
    int y1, bool is_valid) {
  GrayInputTestParam param = {
      .input_dimension = FrameBuffer::Dimension{in_width, in_height},
      .input_format = in_format,
      .output_dimension = FrameBuffer::Dimension{out_width, out_height},
      .output_format = out_format,
      .x0 = x0,
      .y0 = y0,
      .x1 = x1,
      .y1 = y1};
  return std::make_tuple(kCrop, param, is_valid);
}

INSTANTIATE_TEST_SUITE_P(
    ValidateCropInputs, GrayInputTest,
    ::testing::Values(
        CreateGrayCropInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kRGB, 0, 0, 3, 2,
                                     false),
        CreateGrayCropInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kGRAY, 1, 1, 1, 4,
                                     false),
        CreateGrayCropInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 1,
                                     FrameBuffer::Format::kGRAY, -1, 0, 1, 1,
                                     false),
        CreateGrayCropInputTestParam(5, 5, FrameBuffer::Format::kGRAY, 3, 3,
                                     FrameBuffer::Format::kGRAY, 0, 0, 2, 2,
                                     true),
        CreateGrayCropInputTestParam(5, 5, FrameBuffer::Format::kGRAY, 2, 2,
                                     FrameBuffer::Format::kGRAY, 1, 2, 2, 3,
                                     true),
        CreateGrayCropInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 1, 1,
                                     FrameBuffer::Format::kGRAY, 0, 0, 0, 0,
                                     true)));

std::tuple<Operation, GrayInputTestParam, bool> CreateGrayResizeInputTestParam(
    int in_width, int in_height, FrameBuffer::Format in_format, int out_width,
    int out_height, FrameBuffer::Format out_format, bool is_valid) {
  GrayInputTestParam param = {
      .input_dimension = FrameBuffer::Dimension{in_width, in_height},
      .input_format = in_format,
      .output_dimension = FrameBuffer::Dimension{out_width, out_height},
      .output_format = out_format};
  return std::make_tuple(kResize, param, is_valid);
}

INSTANTIATE_TEST_SUITE_P(
    ValidateResizeInputs, GrayInputTest,
    ::testing::Values(
        CreateGrayResizeInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 1, 1,
                                       FrameBuffer::Format::kRGB, false),
        CreateGrayResizeInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 5, 5,
                                       FrameBuffer::Format::kRGB, false),
        CreateGrayResizeInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 2, 1,
                                       FrameBuffer::Format::kGRAY, true),
        CreateGrayResizeInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 7, 9,
                                       FrameBuffer::Format::kGRAY, true)));

std::tuple<Operation, GrayInputTestParam, bool> CreateGrayFlipInputTestParam(
    int in_width, int in_height, FrameBuffer::Format in_format, int out_width,
    int out_height, FrameBuffer::Format out_format, bool horizontal_flip,
    bool is_valid) {
  GrayInputTestParam param = {
      .input_dimension = FrameBuffer::Dimension{in_width, in_height},
      .input_format = in_format,
      .output_dimension = FrameBuffer::Dimension{out_width, out_height},
      .output_format = out_format};
  return std::make_tuple(horizontal_flip ? kHorizontalFlip : kVerticalFlip,
                         param, is_valid);
}

INSTANTIATE_TEST_SUITE_P(
    ValidateFlipInputs, GrayInputTest,
    ::testing::Values(
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kRGB, true, false),
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 3,
                                     FrameBuffer::Format::kGRAY, true, false),
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kGRAY, true, true),
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kRGB, false, false),
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 3,
                                     FrameBuffer::Format::kGRAY, false, false),
        CreateGrayFlipInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                     FrameBuffer::Format::kGRAY, false, true)));

std::tuple<Operation, GrayInputTestParam, bool> CreateGrayConvertInputTestParam(
    int in_width, int in_height, FrameBuffer::Format in_format, int out_width,
    int out_height, FrameBuffer::Format out_format, bool is_valid) {
  GrayInputTestParam param = {
      .input_dimension = FrameBuffer::Dimension{in_width, in_height},
      .input_format = in_format,
      .output_dimension = FrameBuffer::Dimension{out_width, out_height},
      .output_format = out_format};
  return std::make_tuple(kConvert, param, is_valid);
}

INSTANTIATE_TEST_SUITE_P(
    ValidateConvertInputs, GrayInputTest,
    ::testing::Values(
        CreateGrayConvertInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                        FrameBuffer::Format::kRGB, false),
        CreateGrayConvertInputTestParam(3, 2, FrameBuffer::Format::kGRAY, 3, 2,
                                        FrameBuffer::Format::kGRAY, false)));

// Rgb unit tests.
//------------------------------------------------------------------------------

struct FrameBufferPlanarFormat {
  FrameBufferPlanarFormat()
      : format(FrameBuffer::Format::kGRAY), plane_count(0) {}
  FrameBufferPlanarFormat(FrameBuffer::Format format, int plane_count)
      : format(format), plane_count(plane_count) {}

  FrameBuffer::Format format;
  int plane_count;
};

class RgbaConvertTest
    : public testing::TestWithParam<
          std::tuple<FrameBuffer::Format, FrameBufferPlanarFormat>> {
 public:
  void SetUp() override {
    constexpr FrameBuffer::Dimension kBufferDimension = {.width = 2,
                                                         .height = 1};
    constexpr int kBufferSize = 20;
    std::tie(input_format_, output_planar_format_) = GetParam();

    // Setup input frame buffer
    input_data_ = std::make_unique<uint8_t[]>(kBufferSize);
    FrameBuffer::Stride input_stride;
    if (input_format_ == FrameBuffer::Format::kRGBA) {
      uint8_t data[] = {200, 100, 0, 1, 0, 200, 100, 50};
      std::copy(data, data + 8, input_data_.get());
      input_stride = {/*row_stride_bytes=*/8, /*pixel_stride_bytes=*/4};
    } else {
      uint8_t data[] = {200, 100, 0, 0, 200, 100};
      std::copy(data, data + 6, input_data_.get());
      input_stride = {/*row_stride_bytes=*/6, /*pixel_stride_bytes=*/3};
    }
    FrameBuffer::Plane input_plane(/*buffer=*/input_data_.get(), input_stride);
    std::vector<FrameBuffer::Plane> input_planes = {input_plane};
    input_frame_buffer_ = std::make_shared<FrameBuffer>(
        input_planes, kBufferDimension, input_format_);

    // Setup output frame buffer
    if (output_planar_format_.format == FrameBuffer::Format::kRGBA) {
      output_data_1_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_1(
          /*buffer=*/output_data_1_.get(),
          /*stride=*/{/*row_stride_bytes=*/8, /*pixel_stride_bytes=*/4});
      std::vector<FrameBuffer::Plane> output_planes = {output_plane_1};
      output_frame_buffer_ = std::make_shared<FrameBuffer>(
          output_planes, kBufferDimension, output_planar_format_.format);
    } else if (output_planar_format_.format == FrameBuffer::Format::kRGB) {
      output_data_1_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_1(
          /*buffer=*/output_data_1_.get(),
          /*stride=*/{/*row_stride_bytes=*/6, /*pixel_stride_bytes=*/3});
      std::vector<FrameBuffer::Plane> output_planes = {output_plane_1};
      output_frame_buffer_ = std::make_shared<FrameBuffer>(
          output_planes, kBufferDimension, output_planar_format_.format);
    } else if (output_planar_format_.plane_count == 1) {
      output_data_1_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_1(
          /*buffer=*/output_data_1_.get(),
          /*stride=*/{/*row_stride_bytes=*/2,
                      /*pixel_stride_bytes=*/1});
      std::vector<FrameBuffer::Plane> output_planes = {output_plane_1};
      output_frame_buffer_ = std::make_shared<FrameBuffer>(
          output_planes, kBufferDimension, output_planar_format_.format);
    } else if (output_planar_format_.plane_count == 2) {
      output_data_1_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_1(
          /*buffer=*/output_data_1_.get(),
          /*stride=*/{/*row_stride_bytes=*/2, /*pixel_stride_bytes=*/1});
      output_data_2_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_2(
          /*buffer=*/output_data_2_.get(),
          /*stride=*/{/*row_stride_bytes=*/1, /*pixel_stride_bytes=*/2});
      std::vector<FrameBuffer::Plane> planes = {output_plane_1, output_plane_2};
      output_frame_buffer_ = std::make_shared<FrameBuffer>(
          planes, kBufferDimension, output_planar_format_.format);
    } else {
      output_data_1_ = std::make_unique<uint8_t[]>(kBufferSize);
      output_data_2_ = std::make_unique<uint8_t[]>(kBufferSize);
      output_data_3_ = std::make_unique<uint8_t[]>(kBufferSize);
      FrameBuffer::Plane output_plane_1(
          /*buffer=*/output_data_1_.get(),
          /*stride=*/{/*row_stride_bytes=*/2, /*pixel_stride_bytes=*/1});
      FrameBuffer::Plane output_plane_2(
          /*buffer=*/output_data_2_.get(),
          /*stride=*/{/*row_stride_bytes=*/1, /*pixel_stride_bytes=*/1});
      FrameBuffer::Plane output_plane_3(
          /*buffer=*/output_data_3_.get(),
          /*stride=*/{/*row_stride_bytes=*/1, /*pixel_stride_bytes=*/1});
      std::vector<FrameBuffer::Plane> planes = {output_plane_1, output_plane_2,
                                                output_plane_3};
      output_frame_buffer_ = std::make_shared<FrameBuffer>(
          planes, kBufferDimension, output_planar_format_.format);
    }
  }

 protected:
  FrameBuffer::Format input_format_;
  FrameBufferPlanarFormat output_planar_format_;

  std::unique_ptr<uint8_t[]> output_data_1_;
  std::unique_ptr<uint8_t[]> output_data_2_;
  std::unique_ptr<uint8_t[]> output_data_3_;
  std::unique_ptr<uint8_t[]> input_data_;

  std::shared_ptr<FrameBuffer> input_frame_buffer_;
  std::shared_ptr<FrameBuffer> output_frame_buffer_;
};

TEST_P(RgbaConvertTest, RgbaToOtherFormatConversion) {
  absl::Status status =
      Convert(*input_frame_buffer_, output_frame_buffer_.get());
  if (output_planar_format_.format == FrameBuffer::Format::kGRAY) {
    MP_ASSERT_OK(status);
    EXPECT_EQ(output_data_1_[0], 118);
    EXPECT_EQ(output_data_1_[1], 129);
  } else if (output_frame_buffer_->format() == FrameBuffer::Format::kNV12 ||
             output_frame_buffer_->format() == FrameBuffer::Format::kNV21 ||
             output_frame_buffer_->format() == FrameBuffer::Format::kYV12 ||
             output_frame_buffer_->format() == FrameBuffer::Format::kYV21) {
    MP_ASSERT_OK(status);
    MP_ASSERT_OK_AND_ASSIGN(
        FrameBuffer::YuvData yuv_data,
        FrameBuffer::GetYuvDataFromFrameBuffer(*output_frame_buffer_));
    EXPECT_EQ(yuv_data.y_buffer[0], 118);
    EXPECT_EQ(yuv_data.y_buffer[1], 129);
    EXPECT_EQ(yuv_data.u_buffer[0], 61);
    EXPECT_EQ(yuv_data.v_buffer[0], 186);
  } else if (input_format_ == FrameBuffer::Format::kRGBA &&
             output_frame_buffer_->format() == FrameBuffer::Format::kRGB) {
    EXPECT_EQ(output_data_1_[0], 200);
    EXPECT_EQ(output_data_1_[1], 100);
    EXPECT_EQ(output_data_1_[2], 0);
    EXPECT_EQ(output_data_1_[3], 0);
    MP_ASSERT_OK(status);
  } else if (input_format_ == FrameBuffer::Format::kRGB &&
             output_frame_buffer_->format() == FrameBuffer::Format::kRGBA) {
    MP_ASSERT_OK(status);
    EXPECT_EQ(output_data_1_[0], 200);
    EXPECT_EQ(output_data_1_[1], 100);
    EXPECT_EQ(output_data_1_[2], 0);
    EXPECT_EQ(output_data_1_[3], 255);
  } else {
    ASSERT_FALSE(status.ok());
  }
}

INSTANTIATE_TEST_SUITE_P(
    RgbaToOtherFormatConversion, RgbaConvertTest,
    testing::Combine(
        testing::Values(FrameBuffer::Format::kRGBA, FrameBuffer::Format::kRGB),
        testing::Values(FrameBufferPlanarFormat(FrameBuffer::Format::kGRAY,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kRGBA,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kRGB,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV21,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV21,
                                                /*plane_count=*/2),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV21,
                                                /*plane_count=*/3),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV12,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV12,
                                                /*plane_count=*/2),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kNV12,
                                                /*plane_count=*/3),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kYV21,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kYV21,
                                                /*plane_count=*/3),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kYV12,
                                                /*plane_count=*/1),
                        FrameBufferPlanarFormat(FrameBuffer::Format::kYV12,
                                                /*plane_count=*/3))));

TEST(FrameBufferUtil, RgbaToRgbConversion) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 2, .height = 1};
  uint8_t data[] = {200, 100, 0, 1, 0, 200, 100, 50};
  auto input = CreateFromRgbaRawBuffer(data, kBufferDimension);
  uint8_t output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  auto output = CreateFromRgbRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(Convert(*input, output.get()));
  EXPECT_EQ(output_data[0], 200);
  EXPECT_EQ(output_data[1], 100);
  EXPECT_EQ(output_data[2], 0);
  EXPECT_EQ(output_data[3], 0);
  EXPECT_EQ(output_data[4], 200);
  EXPECT_EQ(output_data[5], 100);
}

TEST(FrameBufferUtil, RgbToFloatTensor) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 2, .height = 1};
  constexpr float kScale = 0.1f, kOffset = 0.1f;
  uint8_t data[] = {1, 2, 3, 4, 5, 6};
  auto input = CreateFromRgbRawBuffer(data, kBufferDimension);
  Tensor output(
      Tensor::ElementType::kFloat32,
      Tensor::Shape{1, kBufferDimension.height, kBufferDimension.width, 3});

  MP_ASSERT_OK(ToFloatTensor(*input, kScale, kOffset, output));

  auto view = output.GetCpuReadView();
  const float* output_data = view.buffer<float>();
  EXPECT_EQ(output_data[0], 0.2f);
  EXPECT_EQ(output_data[1], 0.3f);
  EXPECT_EQ(output_data[2], 0.4f);
  EXPECT_EQ(output_data[3], 0.5f);
  EXPECT_EQ(output_data[4], 0.6f);
  EXPECT_EQ(output_data[5], 0.7f);
}

TEST(FrameBufferUtil, RgbaCrop) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kOutputDimension = {.width = 1, .height = 1};
  uint8_t kRgbaTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  uint8_t output_data[4];
  auto input = CreateFromRgbaRawBuffer(kRgbaTestData, kBufferDimension);
  auto output = CreateFromRgbaRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Crop(*input, 0, 1, 0, 1, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 13);
  EXPECT_EQ(output->plane(0).buffer()[1], 14);
  EXPECT_EQ(output->plane(0).buffer()[2], 15);
  EXPECT_EQ(output->plane(0).buffer()[3], 16);
}

TEST(FrameBufferUtil, RgbCrop) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kOutputDimension = {.width = 1, .height = 1};
  uint8_t kRgbTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18};
  uint8_t output_data[3];
  auto input = CreateFromRgbRawBuffer(kRgbTestData, kBufferDimension);
  auto output = CreateFromRgbRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Crop(*input, 0, 1, 0, 1, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 10);
  EXPECT_EQ(output->plane(0).buffer()[1], 11);
  EXPECT_EQ(output->plane(0).buffer()[2], 12);
}

TEST(FrameBufferUtil, RgbaFlipHorizontally) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 1};
  uint8_t kRgbaTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  uint8_t output_data[sizeof(kRgbaTestData) / 2];
  auto input = CreateFromRgbaRawBuffer(kRgbaTestData, kBufferDimension);
  auto output = CreateFromRgbaRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipHorizontally(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 9);
  EXPECT_EQ(output->plane(0).buffer()[1], 10);
  EXPECT_EQ(output->plane(0).buffer()[2], 11);
  EXPECT_EQ(output->plane(0).buffer()[3], 12);
  EXPECT_EQ(output->plane(0).buffer()[4], 5);
  EXPECT_EQ(output->plane(0).buffer()[5], 6);
  EXPECT_EQ(output->plane(0).buffer()[6], 7);
  EXPECT_EQ(output->plane(0).buffer()[7], 8);
  EXPECT_EQ(output->plane(0).buffer()[8], 1);
  EXPECT_EQ(output->plane(0).buffer()[9], 2);
  EXPECT_EQ(output->plane(0).buffer()[10], 3);
  EXPECT_EQ(output->plane(0).buffer()[11], 4);
}

TEST(FrameBufferUtil, RgbFlipHorizontally) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 1};
  uint8_t kRgbTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18};
  uint8_t output_data[sizeof(kRgbTestData) / 2];
  auto input = CreateFromRgbRawBuffer(kRgbTestData, kBufferDimension);
  auto output = CreateFromRgbRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipHorizontally(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 7);
  EXPECT_EQ(output->plane(0).buffer()[1], 8);
  EXPECT_EQ(output->plane(0).buffer()[2], 9);
  EXPECT_EQ(output->plane(0).buffer()[3], 4);
  EXPECT_EQ(output->plane(0).buffer()[4], 5);
  EXPECT_EQ(output->plane(0).buffer()[5], 6);
  EXPECT_EQ(output->plane(0).buffer()[6], 1);
  EXPECT_EQ(output->plane(0).buffer()[7], 2);
  EXPECT_EQ(output->plane(0).buffer()[8], 3);
}

TEST(FrameBufferUtil, RgbaFlipVertically) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2};
  uint8_t kRgbaTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  uint8_t output_data[sizeof(kRgbaTestData)];
  auto input = CreateFromRgbaRawBuffer(kRgbaTestData, kBufferDimension);
  auto output = CreateFromRgbaRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipVertically(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 13);
  EXPECT_EQ(output->plane(0).buffer()[1], 14);
  EXPECT_EQ(output->plane(0).buffer()[2], 15);
  EXPECT_EQ(output->plane(0).buffer()[3], 16);
  EXPECT_EQ(output->plane(0).buffer()[12], 1);
  EXPECT_EQ(output->plane(0).buffer()[13], 2);
  EXPECT_EQ(output->plane(0).buffer()[14], 3);
  EXPECT_EQ(output->plane(0).buffer()[15], 4);
}

TEST(FrameBufferUtil, RgbFlipVertically) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2};
  uint8_t kRgbTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18};
  uint8_t output_data[sizeof(kRgbTestData)];
  auto input = CreateFromRgbRawBuffer(kRgbTestData, kBufferDimension);
  auto output = CreateFromRgbRawBuffer(output_data, kBufferDimension);

  MP_ASSERT_OK(FlipVertically(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 10);
  EXPECT_EQ(output->plane(0).buffer()[1], 11);
  EXPECT_EQ(output->plane(0).buffer()[2], 12);
  EXPECT_EQ(output->plane(0).buffer()[9], 1);
  EXPECT_EQ(output->plane(0).buffer()[10], 2);
  EXPECT_EQ(output->plane(0).buffer()[11], 3);
}

TEST(FrameBufferUtil, RgbaResize) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kResizeUpDimension = {.width = 4,
                                                         .height = 2},
                                   kResizeDownDimension = {.width = 2,
                                                           .height = 2};
  uint8_t kRgbaTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  uint8_t output_data_up[32];
  auto input = CreateFromRgbaRawBuffer(kRgbaTestData, kBufferDimension);
  auto output = CreateFromRgbaRawBuffer(output_data_up, kResizeUpDimension);

  // Test increasing the size.
  MP_ASSERT_OK(Resize(*input, output.get()));
  uint8_t resize_result_size_increase[] = {
      1,  2,  3,  4,  4,  5,  6,  7,  7,  8,  9,  10, 9,  10, 11, 12,
      13, 14, 15, 16, 16, 17, 18, 19, 19, 20, 21, 22, 21, 22, 23, 24};
  for (int i = 0; i < sizeof(output_data_up); i++) {
    EXPECT_EQ(output->plane(0).buffer()[i], resize_result_size_increase[i]);
  }

  // Test shrinking the image by half.
  uint8_t output_data_down[16];
  output = CreateFromRgbaRawBuffer(output_data_down, kResizeDownDimension);

  MP_ASSERT_OK(Resize(*input, output.get()));
  uint8_t resize_result_size_decrease[] = {1,  2,  3,  4,  7,  8,  9,  10,
                                           13, 14, 15, 16, 19, 20, 21, 22};
  for (int i = 0; i < sizeof(output_data_down); i++) {
    EXPECT_EQ(output->plane(0).buffer()[i], resize_result_size_decrease[i]);
  }
}

TEST(FrameBufferUtil, RgbResize) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kResizeUpDimension = {.width = 4,
                                                         .height = 3},
                                   kResizeDownDimension = {.width = 2,
                                                           .height = 2};
  uint8_t kRgbTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18};
  auto input = CreateFromRgbRawBuffer(kRgbTestData, kBufferDimension);

  // Test increasing the size.
  uint8_t output_data_up[36];
  auto output = CreateFromRgbRawBuffer(output_data_up, kResizeUpDimension);
  MP_ASSERT_OK(Resize(*input, output.get()));
  uint8_t resize_result_size_increase[] = {
      1,  2,  3,  3,  4,  5,  5,  6,  7,  7,  8,  9,  7,  8,  9,  9,  10, 11,
      11, 12, 13, 13, 14, 15, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18};
  for (int i = 0; i < sizeof(output_data_up); i++) {
    EXPECT_EQ(output_data_up[i], resize_result_size_increase[i]);
  }

  // Test decreasing the size.
  uint8_t output_data_down[12];
  output = CreateFromRgbRawBuffer(output_data_down, kResizeDownDimension);
  MP_ASSERT_OK(Resize(*input, output.get()));

  uint8_t resize_result_size_decrease[] = {1,  2,  3,  5,  6,  7,
                                           10, 11, 12, 14, 15, 16};
  for (int i = 0; i < sizeof(resize_result_size_decrease); i++) {
    EXPECT_EQ(output->plane(0).buffer()[i], resize_result_size_decrease[i]);
  }
}

TEST(FrameBufferUtil, RgbaRotate) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kRotatedDimension = {.width = 2,
                                                        .height = 3};
  uint8_t kRgbaTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                             13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  uint8_t output_data[sizeof(kRgbaTestData)];
  auto input = CreateFromRgbaRawBuffer(kRgbaTestData, kBufferDimension);
  const std::array<int, 3> kAnglesToTest = {90, 180, 270};
  std::map<int, std::shared_ptr<FrameBuffer>> kOutputBuffers;
  kOutputBuffers[90] = CreateFromRgbaRawBuffer(output_data, kRotatedDimension);
  kOutputBuffers[180] = CreateFromRgbaRawBuffer(output_data, kBufferDimension);
  kOutputBuffers[270] = CreateFromRgbaRawBuffer(output_data, kRotatedDimension);
  const std::map<int, std::array<uint8_t, 24>> kRotationResults{
      {90, {9,  10, 11, 12, 21, 22, 23, 24, 5,  6,  7,  8,
            17, 18, 19, 20, 1,  2,  3,  4,  13, 14, 15, 16}},
      {180, {21, 22, 23, 24, 17, 18, 19, 20, 13, 14, 15, 16,
             9,  10, 11, 12, 5,  6,  7,  8,  1,  2,  3,  4}},
      {270, {13, 14, 15, 16, 1,  2,  3,  4,  17, 18, 19, 20,
             5,  6,  7,  8,  21, 22, 23, 24, 9,  10, 11, 12}}};

  for (auto angle : kAnglesToTest) {
    auto output = kOutputBuffers.at(angle).get();
    MP_ASSERT_OK(Rotate(*input, angle, output));
    auto results = kRotationResults.at(angle);
    for (int i = 0; i < results.size(); i++) {
      EXPECT_EQ(output->plane(0).buffer()[i], results[i]);
    }
  }
}

TEST(FrameBufferUtil, RgbRotate) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 3, .height = 2},
                                   kRotatedDimension = {.width = 2,
                                                        .height = 3};
  uint8_t kRgbTestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18};
  uint8_t output_data[sizeof(kRgbTestData)];
  auto input = CreateFromRgbRawBuffer(kRgbTestData, kBufferDimension);
  const std::array<int, 3> kAnglesToTest = {90, 180, 270};
  std::map<int, std::shared_ptr<FrameBuffer>> kOutputBuffers;
  kOutputBuffers[90] = CreateFromRgbRawBuffer(output_data, kRotatedDimension);
  kOutputBuffers[180] = CreateFromRgbRawBuffer(output_data, kBufferDimension);
  kOutputBuffers[270] = CreateFromRgbRawBuffer(output_data, kRotatedDimension);
  const std::map<int, std::array<uint8_t, 18>> kRotationResults{
      {90, {7, 8, 9, 16, 17, 18, 4, 5, 6, 13, 14, 15, 1, 2, 3, 10, 11, 12}},
      {180, {16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3}},
      {270, {10, 11, 12, 1, 2, 3, 13, 14, 15, 4, 5, 6, 16, 17, 18, 7, 8, 9}}};

  for (auto angle : kAnglesToTest) {
    auto output = kOutputBuffers.at(angle).get();
    MP_ASSERT_OK(Rotate(*input, angle, output));
    auto results = kRotationResults.at(angle);
    for (int i = 0; i < results.size(); i++) {
      EXPECT_EQ(output->plane(0).buffer()[i], results[i]);
    }
  }
}

// Nv21 unit tests.
//------------------------------------------------------------------------------

// Helper function to create YUV buffer.
absl::StatusOr<std::shared_ptr<FrameBuffer>> CreateYuvBuffer(
    uint8_t* buffer, FrameBuffer::Dimension dimension, int plane_count,
    FrameBuffer::Format format) {
  ABSL_DCHECK(plane_count > 0 && plane_count < 4);
  MP_ASSIGN_OR_RETURN(auto uv_dimension,
                      GetUvPlaneDimension(dimension, format));

  if (plane_count == 1) {
    const std::vector<FrameBuffer::Plane> planes = {
        {buffer, /*stride=*/{/*row_stride_bytes=*/dimension.width,
                             /*pixel_stride_bytes=*/1}}};
    return std::make_shared<FrameBuffer>(planes, dimension, format);
  } else if (plane_count == 2) {
    ABSL_CHECK(format == FrameBuffer::Format::kNV12 ||
               format == FrameBuffer::Format::kNV21);
    const std::vector<FrameBuffer::Plane> planes = {
        {buffer,
         /*stride=*/{/*row_stride_bytes=*/dimension.width,
                     /*pixel_stride_bytes=*/1}},
        {buffer + dimension.Size(),
         /*stride=*/{/*row_stride_bytes=*/uv_dimension.width * 2,
                     /*pixel_stride_bytes=*/2}}};
    return std::make_shared<FrameBuffer>(planes, dimension, format);
  } else if (plane_count == 3) {
    std::vector<FrameBuffer::Plane> planes = {
        {buffer,
         /*stride=*/{/*row_stride_bytes=*/dimension.width,
                     /*pixel_stride_bytes=*/1}},
        {buffer + dimension.Size(),
         /*stride=*/{/*row_stride_bytes=*/uv_dimension.width,
                     /*pixel_stride_bytes=*/1}},
        {buffer + dimension.Size() + uv_dimension.Size(),
         /*stride=*/{/*row_stride_bytes=*/uv_dimension.width,
                     /*pixel_stride_bytes=*/1}}};
    return std::make_shared<FrameBuffer>(planes, dimension, format);
  }

  return absl::InvalidArgumentError("The plane_count must between 1 and 3.");
}

TEST(FrameBufferUtil, NV21CreatePlanarYuvBuffer) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2},
                                   kOutputDimension = {.width = 4, .height = 2};
  uint8_t kYTestData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  uint8_t kUTestData[] = {13, 15, 17, 0, 0, 0};
  uint8_t kVTestData[] = {14, 16, 18, 0, 0, 0};
  uint8_t kNV21VUTestData[] = {14, 13, 16, 15, 18, 17};
  const std::vector<FrameBuffer::Plane> three_input_planes = {
      {kYTestData, /*stride=*/{6, 1}},
      {kUTestData, /*stride=*/{3, 1}},
      {kVTestData, /*stride=*/{3, 1}}};
  FrameBuffer three_planar_input(three_input_planes, kBufferDimension,
                                 FrameBuffer::Format::kYV21);

  const std::vector<FrameBuffer::Plane> two_input_planes = {
      {kYTestData, /*stride=*/{6, 1}}, {kNV21VUTestData, /*stride=*/{6, 2}}};
  FrameBuffer two_planar_input(two_input_planes, kBufferDimension,
                               FrameBuffer::Format::kNV21);

  uint8_t output_y[8], output_u[2], output_v[2];
  const std::vector<FrameBuffer::Plane> output_planes = {
      {output_y, /*stride=*/{4, 1}},
      {output_u, /*stride=*/{2, 1}},
      {output_v, /*stride=*/{2, 1}}};
  FrameBuffer output(output_planes, kOutputDimension,
                     FrameBuffer::Format::kYV12);

  MP_ASSERT_OK(Crop(three_planar_input, 2, 0, 5, 1, &output));
  EXPECT_EQ(output.plane(0).buffer()[0], 3);
  EXPECT_EQ(output.plane(0).buffer()[1], 4);
  EXPECT_EQ(output.plane(0).buffer()[2], 5);
  EXPECT_EQ(output.plane(1).buffer()[0], 16);
  EXPECT_EQ(output.plane(2).buffer()[0], 15);

  memset(output_y, 0, sizeof(output_y));
  memset(output_u, 0, sizeof(output_u));
  memset(output_v, 0, sizeof(output_v));
  MP_ASSERT_OK(Crop(two_planar_input, 2, 0, 5, 1, &output));
  EXPECT_EQ(output.plane(0).buffer()[0], 3);
  EXPECT_EQ(output.plane(0).buffer()[1], 4);
  EXPECT_EQ(output.plane(0).buffer()[2], 5);
  EXPECT_EQ(output.plane(1).buffer()[0], 16);
  EXPECT_EQ(output.plane(2).buffer()[0], 15);
}

TEST(FrameBufferUtil, NV21Crop) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2},
                                   kOutputDimension = {.width = 4, .height = 2};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  uint8_t output_data[12];
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(output_data, kOutputDimension,
                                              FrameBuffer::Format::kNV21));

  MP_ASSERT_OK(Crop(*input, 2, 0, 5, 1, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 3);
  EXPECT_EQ(output->plane(0).buffer()[1], 4);
  EXPECT_EQ(output->plane(0).buffer()[2], 5);
  EXPECT_EQ(output->plane(0).buffer()[8], 15);
  EXPECT_EQ(output->plane(0).buffer()[9], 16);
}

TEST(FrameBufferUtil, YV21Crop) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2},
                                   kOutputDimension = {.width = 4, .height = 2};
  uint8_t kYV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 15, 17, 14, 16, 18};
  MP_ASSERT_OK_AND_ASSIGN(
      auto input,
      CreateYuvBuffer(kYV21TestData, kBufferDimension, /*plane_count=*/3,
                      FrameBuffer::Format::kYV21));
  uint8_t output_data[12]{};
  MP_ASSERT_OK_AND_ASSIGN(
      auto output,
      CreateYuvBuffer(output_data, kOutputDimension, /*plane_count=*/3,
                      FrameBuffer::Format::kYV21));

  MP_ASSERT_OK(
      Crop(*input, /*x0=*/2, /*y0=*/0, /*x1=*/5, /*y1=*/1, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 3);
  EXPECT_EQ(output->plane(0).buffer()[1], 4);
  EXPECT_EQ(output->plane(0).buffer()[2], 5);
  EXPECT_EQ(output->plane(1).buffer()[0], 15);
  EXPECT_EQ(output->plane(2).buffer()[0], 16);
}

TEST(FrameBufferUtil, NV21HorizontalFlip) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  uint8_t output_data[18];
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(output_data, kBufferDimension,
                                              FrameBuffer::Format::kNV21));

  MP_ASSERT_OK(FlipHorizontally(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 6);
  EXPECT_EQ(output->plane(0).buffer()[1], 5);
  EXPECT_EQ(output->plane(0).buffer()[2], 4);
  EXPECT_EQ(output->plane(0).buffer()[12], 17);
  EXPECT_EQ(output->plane(0).buffer()[13], 18);
}

TEST(FrameBufferUtil, NV21VerticalFlip) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  uint8_t output_data[18];
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(output_data, kBufferDimension,
                                              FrameBuffer::Format::kNV21));

  MP_ASSERT_OK(FlipVertically(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 7);
  EXPECT_EQ(output->plane(0).buffer()[1], 8);
  EXPECT_EQ(output->plane(0).buffer()[2], 9);
  EXPECT_EQ(output->plane(0).buffer()[12], 13);
  EXPECT_EQ(output->plane(0).buffer()[13], 14);
}

TEST(FrameBufferUtil, NV21Rotate) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2},
                                   kRotatedDimension = {.width = 2,
                                                        .height = 6};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  uint8_t output_data[18];
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(output_data, kRotatedDimension,
                                              FrameBuffer::Format::kNV21));

  MP_ASSERT_OK(Rotate(*input, 90, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 6);
  EXPECT_EQ(output->plane(0).buffer()[1], 12);
  EXPECT_EQ(output->plane(0).buffer()[2], 5);
  EXPECT_EQ(output->plane(0).buffer()[12], 17);
  EXPECT_EQ(output->plane(0).buffer()[13], 18);
}

TEST(FrameBufferUtil, NV21Resize) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2},
                                   kOutputDimension = {.width = 1, .height = 1};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  uint8_t output_data[6];
  MP_ASSERT_OK_AND_ASSIGN(auto output,
                          CreateFromRawBuffer(output_data, kOutputDimension,
                                              FrameBuffer::Format::kNV21));

  MP_ASSERT_OK(Resize(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 1);
  EXPECT_EQ(output->plane(0).buffer()[1], 13);
}

TEST(FrameBufferUtil, NV21ConvertGray) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(auto input,
                          CreateFromRawBuffer(kNV21TestData, kBufferDimension,
                                              FrameBuffer::Format::kNV21));
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kGRAY);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromGrayRawBuffer(output_data.data(), kBufferDimension);

  MP_ASSERT_OK(Convert(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 1);
  EXPECT_EQ(output->plane(0).buffer()[1], 2);
  EXPECT_EQ(output->plane(0).buffer()[11], 12);
}

TEST(FrameBufferUtil, PaddedYuvConvertGray) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kNV21PaddedTestData[] = {1,  2,  3,  4,  5,  6,  100, 100,
                                   7,  8,  9,  10, 11, 12, 100, 100,
                                   13, 14, 15, 16, 17, 18, 100, 100};
  constexpr int row_stride_y = 8;
  const std::vector<FrameBuffer::Plane> planes = {
      {kNV21PaddedTestData, /*stride=*/{row_stride_y, 1}},
      {kNV21PaddedTestData + (row_stride_y * kBufferDimension.width),
       /*stride=*/{row_stride_y, 2}}};
  auto input = std::make_shared<FrameBuffer>(planes, kBufferDimension,
                                             FrameBuffer::Format::kNV21);
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kGRAY);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromGrayRawBuffer(output_data.data(), kBufferDimension);

  MP_ASSERT_OK(Convert(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 1);
  EXPECT_EQ(output->plane(0).buffer()[1], 2);
  EXPECT_EQ(output->plane(0).buffer()[6], 7);
  EXPECT_EQ(output->plane(0).buffer()[7], 8);
  EXPECT_EQ(output->plane(0).buffer()[11], 12);
}

TEST(FrameBufferUtil, NV21ConvertRgb) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 32,
                                                       .height = 8};
  // Note that RGB conversion expects at images width at least width >= 32
  // because the implementation is vectorized.
  const int kInputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kNV21);
  std::vector<uint8_t> input_data(kInputSize);
  input_data.data()[0] = 1;
  input_data.data()[1] = 2;
  input_data.data()[32] = 7;
  input_data.data()[33] = 8;
  input_data.data()[256] = 13;
  input_data.data()[257] = 14;
  MP_ASSERT_OK_AND_ASSIGN(
      auto input, CreateFromRawBuffer(input_data.data(), kBufferDimension,
                                      FrameBuffer::Format::kNV21));
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kRGB);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromRgbRawBuffer(output_data.data(), kBufferDimension);

  MP_ASSERT_OK(Convert(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 0);
  EXPECT_EQ(output->plane(0).buffer()[1], 122);
}

TEST(FrameBufferUtil, NV21ConvertHalfRgb) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 64,
                                                       .height = 16},
                                   kOutputDimension = {.width = 32,
                                                       .height = 8};
  // Note that RGB conversion expects at images width at least width >= 32
  // because the implementation is vectorized.
  uint8_t data[1576];
  for (int i = 0; i < sizeof(data); i++) {
    data[i] = (i + 1);
  }
  MP_ASSERT_OK_AND_ASSIGN(
      auto input,
      CreateFromRawBuffer(data, kBufferDimension, FrameBuffer::Format::kNV21));
  uint8_t output_data[768];
  auto output = CreateFromRgbRawBuffer(output_data, kOutputDimension);

  MP_ASSERT_OK(Convert(*input, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 0);
  EXPECT_EQ(output->plane(0).buffer()[1], 135);
}

TEST(FrameBufferUtil, NV12ConvertGray) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kYTestData[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  uint8_t kNV12UVTestData[] = {13, 14, 15, 16, 17, 18};
  const std::vector<FrameBuffer::Plane> planes_nv12 = {
      {kYTestData, /*stride=*/{kBufferDimension.width, 1}},
      {kNV12UVTestData, /*stride=*/{kBufferDimension.width, 2}}};
  auto buffer_nv12 = std::make_shared<FrameBuffer>(
      planes_nv12, kBufferDimension, FrameBuffer::Format::kNV12);
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kGRAY);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromGrayRawBuffer(output_data.data(), kBufferDimension);

  MP_ASSERT_OK(Convert(*buffer_nv12, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], kYTestData[0]);
  EXPECT_EQ(output->plane(0).buffer()[1], kYTestData[1]);
  EXPECT_EQ(output->plane(0).buffer()[11], kYTestData[11]);
}

TEST(FrameBufferUtil, NV12ConvertRgb) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 32,
                                                       .height = 8};
  MP_ASSERT_OK_AND_ASSIGN(
      FrameBuffer::Dimension uv_dimension,
      GetUvPlaneDimension(kBufferDimension, FrameBuffer::Format::kNV12));
  // Halide RGB converter expects at images width at least width >= 32 because
  // the implementation is vectorized.
  auto y_data = std::make_unique<uint8_t[]>(kBufferDimension.Size());
  auto uv_data = std::make_unique<uint8_t[]>(uv_dimension.Size() * 2);
  y_data[0] = 1;
  y_data[1] = 2;
  y_data[32] = 7;
  y_data[33] = 8;
  uv_data[0] = 13;
  uv_data[1] = 14;
  const std::vector<FrameBuffer::Plane> planes_nv12 = {
      {y_data.get(), /*stride=*/{kBufferDimension.width, 1}},
      {uv_data.get(), /*stride=*/{kBufferDimension.width, 2}}};
  auto buffer_nv12 = std::make_shared<FrameBuffer>(
      planes_nv12, kBufferDimension, FrameBuffer::Format::kNV12);
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kRGB);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromRgbRawBuffer(output_data.data(), kBufferDimension);

  MP_ASSERT_OK(Convert(*buffer_nv12, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 0);
  EXPECT_EQ(output->plane(0).buffer()[1], 122);
}

TEST(FrameBufferUtil, NV12ConvertHalfRgb) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 64,
                                                       .height = 16};
  MP_ASSERT_OK_AND_ASSIGN(
      FrameBuffer::Dimension uv_dimension,
      GetUvPlaneDimension(kBufferDimension, FrameBuffer::Format::kNV12));
  // Halide RGB converter expects at images width at least width >= 32 because
  // the implementation is vectorized.
  auto y_data = std::make_unique<uint8_t[]>(kBufferDimension.Size());
  auto uv_data = std::make_unique<uint8_t[]>(uv_dimension.Size() * 2);
  for (int i = 0; i < kBufferDimension.Size(); i++) {
    y_data[i] = (i + 1) % 256;
  }
  for (int i = 0; i < uv_dimension.Size() * 2; i++) {
    uv_data[i] = (i + 1) % 256;
  }
  const std::vector<FrameBuffer::Plane> planes_nv12 = {
      {y_data.get(), /*stride=*/{kBufferDimension.width, 1}},
      {uv_data.get(), /*stride=*/{kBufferDimension.width, 2}}};
  auto buffer_nv12 = std::make_shared<FrameBuffer>(
      planes_nv12, kBufferDimension, FrameBuffer::Format::kNV12);
  constexpr FrameBuffer::Dimension kOutputDimension = {
      .width = kBufferDimension.width / 2,
      .height = kBufferDimension.height / 2};
  const int kOutputSize =
      GetFrameBufferByteSize(kOutputDimension, FrameBuffer::Format::kRGB);
  std::vector<uint8_t> output_data(kOutputSize);
  auto output = CreateFromRgbRawBuffer(output_data.data(), kOutputDimension);

  MP_ASSERT_OK(Convert(*buffer_nv12, output.get()));
  EXPECT_EQ(output->plane(0).buffer()[0], 0);
  EXPECT_EQ(output->plane(0).buffer()[1], 135);
}

TEST(FrameBufferUtil, NV21ConvertYV12) {
  constexpr FrameBuffer::Dimension kBufferDimension = {.width = 6, .height = 2};
  uint8_t kNV21TestData[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18};
  MP_ASSERT_OK_AND_ASSIGN(
      auto nv21,
      CreateYuvBuffer(kNV21TestData, kBufferDimension, /*plane_count=*/2,
                      FrameBuffer::Format::kNV21));
  MP_ASSERT_OK_AND_ASSIGN(FrameBuffer::YuvData nv21_data,
                          FrameBuffer::GetYuvDataFromFrameBuffer(*nv21));
  const int kOutputSize =
      GetFrameBufferByteSize(kBufferDimension, FrameBuffer::Format::kYV12);
  std::vector<uint8_t> output_data(kOutputSize);
  MP_ASSERT_OK_AND_ASSIGN(
      auto yv12,
      CreateYuvBuffer(output_data.data(), kBufferDimension, /*plane_count=*/3,
                      FrameBuffer::Format::kYV12));
  MP_ASSERT_OK_AND_ASSIGN(FrameBuffer::YuvData yv12_data,
                          FrameBuffer::GetYuvDataFromFrameBuffer(*yv12));

  MP_ASSERT_OK(Convert(*nv21, yv12.get()));
  EXPECT_EQ(nv21_data.y_buffer[0], yv12_data.y_buffer[0]);
  EXPECT_EQ(nv21_data.u_buffer[0], yv12_data.u_buffer[0]);
  EXPECT_EQ(nv21_data.v_buffer[0], yv12_data.v_buffer[0]);
}

}  // namespace
}  // namespace frame_buffer
}  // namespace mediapipe
