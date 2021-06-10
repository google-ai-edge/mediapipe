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

#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace mediapipe {

namespace tf = tensorflow;
using RandomEngine = std::mt19937_64;

const uint8 kGray8 = 42;
const uint16 kGray16 = 4242;
const float kFloat = 42.0;
const uint kRed = 255;
const uint kGreen = 36;
const uint kBlue = 156;
const uint kAlpha = 42;

const int kFixedNoiseWidth = 3;
const int kFixedNoiseHeight = 2;
const uint8 kFixedNoiseData[kFixedNoiseWidth * kFixedNoiseHeight * 3] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 123, 213, 156, 9, 10, 11, 255, 0, 128};

class ImageFrameToTensorCalculatorTest : public ::testing::Test {
 protected:
  // Set image_frame to a constant per-channel pix_value.
  template <class T>
  void SetToColor(const T* pix_value, ImageFrame* image_frame) {
    const int cols = image_frame->Width();
    const int rows = image_frame->Height();
    const int channels = image_frame->NumberOfChannels();
    const int width_padding =
        image_frame->WidthStep() / (sizeof(T)) - cols * channels;
    T* pixel = reinterpret_cast<T*>(image_frame->MutablePixelData());
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        for (int channel = 0; channel < channels; ++channel) {
          pixel[channel] = pix_value[channel];
        }
        pixel += channels;
      }
      pixel += width_padding;
    }
  }

  // Adds a packet with a solid red 8-bit RGB ImageFrame.
  void AddRGBFrame(int width, int height) {
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, width, height);
    const uint8 color[] = {kRed, kGreen, kBlue};
    SetToColor<uint8>(color, image_frame.get());
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with a solid red 8-bit RGBA ImageFrame.
  void AddRGBAFrame(int width, int height) {
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::SRGBA, width, height);
    const uint8 color[] = {kRed, kGreen, kBlue, kAlpha};
    SetToColor<uint8>(color, image_frame.get());
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with a solid GRAY8 ImageFrame.
  void AddGray8Frame(int width, int height) {
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::GRAY8, width, height);
    const uint8 gray[] = {kGray8};
    SetToColor<uint8>(gray, image_frame.get());
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with a solid GRAY16 ImageFrame.
  void AddGray16Frame(int width, int height) {
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::GRAY16, width, height, 1);
    const uint16 gray[] = {kGray16};
    SetToColor<uint16>(gray, image_frame.get());
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with a solid VEC32F1 ImageFrame.
  void AddFloatFrame(int width, int height) {
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::VEC32F1, width, height, 1);
    const float gray[] = {kFloat};
    SetToColor<float>(gray, image_frame.get());
    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with an 8-bit RGB ImageFrame containing pre-determined noise.
  void AddFixedNoiseRGBFrame() {
    auto image_frame = ::absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, kFixedNoiseWidth, kFixedNoiseHeight);

    // Copy fixed noise data into the ImageFrame.
    const uint8* src = kFixedNoiseData;
    uint8* pixels = image_frame->MutablePixelData();
    for (int y = 0; y < kFixedNoiseHeight; ++y) {
      uint8* row = pixels + y * image_frame->WidthStep();
      std::memcpy(row, src, kFixedNoiseWidth * 3);
      src += kFixedNoiseWidth * 3;
    }

    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  // Adds a packet with an 8-bit RGB ImageFrame containing random noise.
  void AddRandomRGBFrame(int width, int height, uint32 seed) {
    RandomEngine random(seed);
    std::uniform_int_distribution<int> uniform_dist{
        0, std::numeric_limits<uint8_t>::max()};
    auto image_frame =
        ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, width, height);

    // Copy "noisy data" into the ImageFrame.
    const int num_components_per_row = width * image_frame->NumberOfChannels();
    uint8* pixels = image_frame->MutablePixelData();
    for (int y = 0; y < kFixedNoiseHeight; ++y) {
      uint8* p = pixels + y * image_frame->WidthStep();
      for (int i = 0; i < num_components_per_row; ++i) {
        p[i] = uniform_dist(random);
      }
    }

    runner_->MutableInputs()->Index(0).packets.push_back(
        Adopt(image_frame.release()).At(Timestamp(0)));
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(ImageFrameToTensorCalculatorTest, SolidRedRGBFrame) {
  // Check two widths to cover packed and padded ImageFrame.
  const int num_widths = 2;
  const int widths[num_widths] = {10, 24};
  const int height = 5;
  for (int width_index = 0; width_index < num_widths; ++width_index) {
    const int width = widths[width_index];
    const int num_pixels = width * height;

    // Run the calculator and verify that one output is generated.
    runner_ = ::absl::make_unique<CalculatorRunner>(
        "ImageFrameToTensorCalculator", "", 1, 1, 0);
    AddRGBFrame(width, height);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_UINT8, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(3, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    const uint8* pixels =
        reinterpret_cast<const uint8*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      ASSERT_EQ(kRed, pixels[0]);
      ASSERT_EQ(kGreen, pixels[1]);
      ASSERT_EQ(kBlue, pixels[2]);
      pixels += 3;
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, SolidRedRGBAFrame) {
  // Check two widths to cover packed and padded ImageFrame.
  const int num_widths = 2;
  const int widths[num_widths] = {10, 24};
  const int height = 5;
  for (int width_index = 0; width_index < num_widths; ++width_index) {
    const int width = widths[width_index];
    const int num_pixels = width * height;

    // Run the calculator and verify that one output is generated.
    runner_.reset(
        new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
    AddRGBAFrame(width, height);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_UINT8, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(4, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    const uint8* pixels =
        reinterpret_cast<const uint8*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      ASSERT_EQ(kRed, pixels[0]);
      ASSERT_EQ(kGreen, pixels[1]);
      ASSERT_EQ(kBlue, pixels[2]);
      ASSERT_EQ(kAlpha, pixels[3]);
      pixels += 4;
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, SolidGray8Frame) {
  // Check two widths to cover packed and padded ImageFrame.
  const int num_widths = 2;
  const int widths[num_widths] = {10, 24};
  const int height = 5;
  for (int width_index = 0; width_index < num_widths; ++width_index) {
    const int width = widths[width_index];
    const int num_pixels = width * height;

    // Run the calculator and verify that one output is generated.
    runner_.reset(
        new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
    AddGray8Frame(width, height);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_UINT8, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(1, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    const uint8* pixels =
        reinterpret_cast<const uint8*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      ASSERT_EQ(kGray8, pixels[0]);
      ++pixels;
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, SolidGray16Frame) {
  // Check two widths to cover packed and padded ImageFrame.
  const int num_widths = 2;
  const int widths[num_widths] = {10, 24};
  const int height = 5;
  for (int width_index = 0; width_index < num_widths; ++width_index) {
    const int width = widths[width_index];
    const int num_pixels = width * height;

    // Run the calculator and verify that one output is generated.
    runner_.reset(
        new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
    AddGray16Frame(width, height);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_UINT16, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(1, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    const uint16* pixels =
        reinterpret_cast<const uint16*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      ASSERT_EQ(kGray16, pixels[0]);
      ++pixels;
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, SolidFloatFrame) {
  // Check two widths to cover packed and padded ImageFrame.
  const int num_widths = 2;
  const int widths[num_widths] = {10, 24};
  const int height = 5;
  for (int width_index = 0; width_index < num_widths; ++width_index) {
    const int width = widths[width_index];
    const int num_pixels = width * height;

    // Run the calculator and verify that one output is generated.
    runner_.reset(
        new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
    AddFloatFrame(width, height);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_FLOAT, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(1, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    const float* pixels =
        reinterpret_cast<const float*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      ASSERT_EQ(kFloat, pixels[0]);
      ++pixels;
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, FixedNoiseRGBFrame) {
  // Run the calculator and verify that one output is generated.
  runner_.reset(
      new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
  AddFixedNoiseRGBFrame();
  MP_ASSERT_OK(runner_->Run());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Index(0).packets;
  ASSERT_EQ(1, output_packets.size());

  // Verify that tensor is 3-dimensional
  const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
  ASSERT_EQ(3, tensor.dims());
  ASSERT_EQ(tf::DT_UINT8, tensor.dtype());

  // Verify that each dimension has the correct size / number of channels.
  const tf::TensorShape& shape = tensor.shape();
  ASSERT_EQ(kFixedNoiseHeight, shape.dim_size(0));
  ASSERT_EQ(kFixedNoiseWidth, shape.dim_size(1));
  ASSERT_EQ(3, shape.dim_size(2));

  // Verify that the data in the tensor is correct.
  const int num_pixels = kFixedNoiseWidth * kFixedNoiseHeight;
  const uint8* pixels =
      reinterpret_cast<const uint8*>(tensor.tensor_data().data());
  for (int i = 0; i < num_pixels; ++i) {
    ASSERT_EQ(kFixedNoiseData[i], pixels[i]);
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, RandomRGBFrame) {
  // Run the calculator and verify that one output is generated.
  const uint32 seed = 1234;
  const int height = 2;
  for (int width = 1; width <= 33; ++width) {
    runner_.reset(
        new CalculatorRunner("ImageFrameToTensorCalculator", "", 1, 1, 0));
    AddRandomRGBFrame(width, height, seed);
    MP_ASSERT_OK(runner_->Run());
    const std::vector<Packet>& output_packets =
        runner_->Outputs().Index(0).packets;
    ASSERT_EQ(1, output_packets.size());

    // Verify that tensor is 3-dimensional
    const tf::Tensor& tensor = output_packets[0].Get<tf::Tensor>();
    ASSERT_EQ(3, tensor.dims());
    ASSERT_EQ(tf::DT_UINT8, tensor.dtype());

    // Verify that each dimension has the correct size / number of channels.
    const tf::TensorShape& shape = tensor.shape();
    ASSERT_EQ(height, shape.dim_size(0));
    ASSERT_EQ(width, shape.dim_size(1));
    ASSERT_EQ(3, shape.dim_size(2));

    // Verify that the data in the tensor is correct.
    RandomEngine random(seed);
    std::uniform_int_distribution<int> uniform_dist{
        0, std::numeric_limits<uint8_t>::max()};
    const int num_pixels = width * height;
    const uint8* pixels =
        reinterpret_cast<const uint8*>(tensor.tensor_data().data());
    for (int i = 0; i < num_pixels; ++i) {
      const uint8 expected = uniform_dist(random);
      ASSERT_EQ(expected, pixels[i]);
    }
  }
}

TEST_F(ImageFrameToTensorCalculatorTest, FixedRGBFrameWithMeanAndStddev) {
  runner_ = ::absl::make_unique<CalculatorRunner>(
      "ImageFrameToTensorCalculator",
      "[mediapipe.ImageFrameToTensorCalculatorOptions.ext]"
      "{data_type:DT_FLOAT mean:128.0 stddev:128.0}",
      1, 1, 0);

  // Create a single pixel image of fixed color #0080ff.
  auto image_frame = ::absl::make_unique<ImageFrame>(ImageFormat::SRGB, 1, 1);
  const uint8 color[] = {0, 128, 255};
  SetToColor<uint8>(color, image_frame.get());

  runner_->MutableInputs()->Index(0).packets.push_back(
      Adopt(image_frame.release()).At(Timestamp(0)));
  MP_ASSERT_OK(runner_->Run());

  const auto& tensor = runner_->Outputs().Index(0).packets[0].Get<tf::Tensor>();
  EXPECT_EQ(tensor.dtype(), tf::DT_FLOAT);
  ASSERT_EQ(tensor.dims(), 3);
  EXPECT_EQ(tensor.shape().dim_size(0), 1);
  EXPECT_EQ(tensor.shape().dim_size(1), 1);
  EXPECT_EQ(tensor.shape().dim_size(2), 3);
  const float* actual = tensor.flat<float>().data();
  EXPECT_EQ(actual[0], -1.0f);            // (  0 - 128) / 128
  EXPECT_EQ(actual[1], 0.0f);             // (128 - 128) / 128
  EXPECT_EQ(actual[2], 127.0f / 128.0f);  // (255 - 128) / 128
}

}  // namespace mediapipe
