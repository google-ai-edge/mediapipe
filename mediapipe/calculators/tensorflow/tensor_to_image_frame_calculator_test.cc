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
#include <type_traits>

#include "mediapipe/calculators/tensorflow/tensor_to_image_frame_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

constexpr char kTensor[] = "TENSOR";
constexpr char kImage[] = "IMAGE";

}  // namespace

template <class TypeParam>
class TensorToImageFrameCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner(bool scale_per_frame_min_max = false) {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToImageFrameCalculator");
    config.add_input_stream("TENSOR:input_tensor");
    config.add_output_stream("IMAGE:output_image");
    config.mutable_options()
        ->MutableExtension(mediapipe::TensorToImageFrameCalculatorOptions::ext)
        ->set_scale_per_frame_min_max(scale_per_frame_min_max);
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

using TensorToImageFrameCalculatorTestTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(TensorToImageFrameCalculatorTest,
                TensorToImageFrameCalculatorTestTypes);

TYPED_TEST(TensorToImageFrameCalculatorTest, Converts3DTensorToImageFrame) {
  // TYPED_TEST requires explicit "this->"
  this->SetUpRunner();
  auto& runner = this->runner_;
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape{kHeight, kWidth, 3};
  auto tensor = absl::make_unique<tf::Tensor>(
      tf::DataTypeToEnum<TypeParam>::v(), tensor_shape);
  auto tensor_vec = tensor->template flat<TypeParam>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight * 3; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64_t time = 1234;
  runner->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner->Run().ok());
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(ImageFormat::SRGB, output_image.Format());
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight * 3; ++i) {
    const uint8_t pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

TYPED_TEST(TensorToImageFrameCalculatorTest, Converts3DTensorToImageFrameGray) {
  this->SetUpRunner();
  auto& runner = this->runner_;
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape{kHeight, kWidth, 1};
  auto tensor = absl::make_unique<tf::Tensor>(
      tf::DataTypeToEnum<TypeParam>::v(), tensor_shape);
  auto tensor_vec = tensor->template flat<TypeParam>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64_t time = 1234;
  runner->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner->Run().ok());
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(ImageFormat::GRAY8, output_image.Format());
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight; ++i) {
    const uint8_t pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

TYPED_TEST(TensorToImageFrameCalculatorTest,
           Converts3DTensorToImageFrame2DGray) {
  this->SetUpRunner();
  auto& runner = this->runner_;
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape{kHeight, kWidth};
  auto tensor = absl::make_unique<tf::Tensor>(
      tf::DataTypeToEnum<TypeParam>::v(), tensor_shape);
  auto tensor_vec = tensor->template flat<TypeParam>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64_t time = 1234;
  runner->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner->Run().ok());
  const std::vector<Packet>& output_packets =
      runner->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(ImageFormat::GRAY8, output_image.Format());
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight; ++i) {
    const uint8_t pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

TYPED_TEST(TensorToImageFrameCalculatorTest,
           Converts3DTensorToImageFrame2DGrayWithScaling) {
  this->SetUpRunner(true);
  auto& runner = this->runner_;
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape{kHeight, kWidth};
  auto tensor = absl::make_unique<tf::Tensor>(
      tf::DataTypeToEnum<TypeParam>::v(), tensor_shape);
  auto tensor_vec = tensor->template flat<TypeParam>().data();

  // Writing sequence of integers as floats which we want normalized.
  tensor_vec[0] = 255;
  for (int i = 1; i < kWidth * kHeight; ++i) {
    tensor_vec[i] = 200;
  }

  const int64_t time = 1234;
  runner->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  if (!std::is_same<TypeParam, float>::value) {
    EXPECT_FALSE(runner->Run().ok());
    return;  // Short circuit because does not apply to other types.
  } else {
    EXPECT_TRUE(runner->Run().ok());
    const std::vector<Packet>& output_packets =
        runner->Outputs().Tag(kImage).packets;
    EXPECT_EQ(1, output_packets.size());
    EXPECT_EQ(time, output_packets[0].Timestamp().Value());
    const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
    EXPECT_EQ(ImageFormat::GRAY8, output_image.Format());
    EXPECT_EQ(kWidth, output_image.Width());
    EXPECT_EQ(kHeight, output_image.Height());

    EXPECT_EQ(255, output_image.PixelData()[0]);
    for (int i = 1; i < kWidth * kHeight; ++i) {
      const uint8_t pixel_value = output_image.PixelData()[i];
      ASSERT_EQ(0, pixel_value);
    }
  }
}

}  // namespace mediapipe
