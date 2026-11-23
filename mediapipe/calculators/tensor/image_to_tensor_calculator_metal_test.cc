// Copyright 2026 The MediaPipe Authors.
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

// Tests ImageToTensorCalculator with GPU input images on Apple platforms, i.e.
// the Metal converter.
//
// What this tests:
// - 4-channel output (the default, written by the render pipeline) and packed
//   3-channel output (`metal_three_channel_output`, written by the compute
//   kernel).
// - float32, uint8 and int8 output tensors (selected by the output range).
// - Image sizes whose 4-channel output rows do and don't meet the linear
//   texture alignment, i.e. the converter rendering directly into the output
//   buffer or into a separate texture that it blits into the buffer (see the
//   test parameters).
// - Exact values: the full image is converted at its own resolution, so each
//   output pixel samples exactly one input pixel, and the expected values are
//   the input pixels mapped to the output range.
// - Clamping of 8-bit outputs: float16 input textures can hold values outside
//   [0, 1], which are mapped outside the output range and must be clamped.
// - The ROI transform and both border modes: an ROI shifted by one pixel, so
//   that the first output column samples outside the image.
//
// What this does not test:
// - On the iOS Simulator, the converter always renders 4-channel output into a
//   separate texture and blits it into the output buffer, since the Simulator
//   doesn't support linear textures backed by shared MTLBuffers. Rendering
//   directly into the output buffer is only covered when running on a device.
// - Resampling (scaling, rotation, keep_aspect_ratio) and float16 output (not
//   used by ImageToTensorCalculator).
//
// The test is empty on platforms without Metal.

#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED
#include <CoreVideo/CoreVideo.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "mediapipe/calculators/tensor/image_to_tensor_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/framework/tool/sink.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/objc/CFHolder.h"
#include "mediapipe/objc/util.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

using ::absl_testing::IsOkAndHolds;

using ::testing::FloatNear;
using ::testing::Pointwise;

struct TestParams {
  std::string name;
  int num_channels;
  int width;
  int height;
};

class ImageToTensorCalculatorMetalTest
    : public testing::TestWithParam<TestParams> {
 protected:
  // Converts the full `image` (by default the test RGBA image) at its own
  // resolution (no resampling) with the given output range (which selects the
  // output tensor type) and returns the tensor's elements. The ROI is shifted
  // horizontally by `roi_x_shift_pixels`.
  template <typename T>
  absl::StatusOr<std::vector<T>> Convert(
      Tensor::ElementType expected_type, ImageToTensorCalculatorOptions options,
      std::optional<GpuBuffer> image = std::nullopt,
      float roi_x_shift_pixels = 0.0f) {
    const TestParams& params = GetParam();
    if (!image.has_value()) {
      ABSL_ASSIGN_OR_RETURN(
          CFHolder<CVPixelBufferRef> pixel_buffer,
          CreateCVPixelBufferForImageFrame(std::make_shared<ImageFrame>(
              CreateTestRgba8ImageFrame(params.width, params.height))));
      image = GpuBuffer(pixel_buffer);
    }
    // Built in code since iOS uses lite protos (no text format).
    CalculatorGraphConfig config;
    config.add_input_stream("image_gpu");
    config.add_input_stream("norm_rect");
    CalculatorGraphConfig::Node* to_tensor = config.add_node();
    to_tensor->set_calculator("ImageToTensorCalculator");
    to_tensor->add_input_stream("IMAGE_GPU:image_gpu");
    to_tensor->add_input_stream("NORM_RECT:norm_rect");
    to_tensor->add_output_stream("TENSORS:tensors");
    options.set_output_tensor_width(params.width);
    options.set_output_tensor_height(params.height);
    if (params.num_channels == 3) {
      options.set_metal_three_channel_output(true);
    }
    *to_tensor->mutable_options()->MutableExtension(
        ImageToTensorCalculatorOptions::ext) = options;
    std::vector<Packet> output_packets;
    tool::AddVectorSink("tensors", &config, &output_packets);

    CalculatorGraph graph;
    ABSL_RETURN_IF_ERROR(graph.Initialize(config));
    ABSL_RETURN_IF_ERROR(graph.StartRun({}));
    NormalizedRect rect;
    rect.set_x_center(0.5f + roi_x_shift_pixels / params.width);
    rect.set_y_center(0.5f);
    rect.set_width(1.0f);
    rect.set_height(1.0f);
    ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "image_gpu", MakePacket<GpuBuffer>(*image).At(Timestamp(0))));
    ABSL_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "norm_rect", MakePacket<NormalizedRect>(rect).At(Timestamp(0))));
    ABSL_RETURN_IF_ERROR(graph.WaitUntilIdle());

    RET_CHECK_EQ(output_packets.size(), 1);
    const auto& tensors = output_packets[0].Get<std::vector<Tensor>>();
    RET_CHECK_EQ(tensors.size(), 1);
    const Tensor& tensor = tensors[0];
    RET_CHECK_EQ(Tensor::ElementTypeName(tensor.element_type()),
                 Tensor::ElementTypeName(expected_type));
    RET_CHECK(tensor.shape().dims ==
              std::vector<int>(
                  {1, params.height, params.width, params.num_channels}));
    std::vector<T> result;
    {
      // Waits for the Metal command buffer that writes the tensor.
      auto view = tensor.GetCpuReadView();
      const T* data = view.buffer<T>();
      result.assign(data, data + tensor.shape().num_elements());
    }
    output_packets.clear();
    ABSL_RETURN_IF_ERROR(graph.CloseAllInputStreams());
    ABSL_RETURN_IF_ERROR(graph.WaitUntilDone());
    return result;
  }

  // Returns the expected tensor elements: the test image's RGB values plus
  // `offset`, and (for 4 channels) an alpha of 0.
  template <typename T>
  std::vector<T> Expected(int offset = 0) {
    const TestParams& params = GetParam();
    ImageFrame frame = CreateTestRgba8ImageFrame(params.width, params.height);
    std::vector<T> expected;
    for (int y = 0; y < params.height; ++y) {
      const uint8_t* row = frame.PixelData() + y * frame.WidthStep();
      for (int x = 0; x < params.width; ++x) {
        for (int c = 0; c < 3; ++c) {
          expected.push_back(static_cast<T>(row[4 * x + c] + offset));
        }
        if (params.num_channels == 4) {
          expected.push_back(0);
        }
      }
    }
    return expected;
  }
};

TEST_P(ImageToTensorCalculatorMetalTest, Float32Output) {
  ImageToTensorCalculatorOptions options;
  options.mutable_output_tensor_float_range()->set_min(0.0f);
  options.mutable_output_tensor_float_range()->set_max(255.0f);
  EXPECT_THAT(Convert<float>(Tensor::ElementType::kFloat32, options),
              IsOkAndHolds(Pointwise(FloatNear(1e-3f), Expected<float>())));
}

TEST_P(ImageToTensorCalculatorMetalTest, UInt8Output) {
  ImageToTensorCalculatorOptions options;
  options.mutable_output_tensor_uint_range()->set_min(0);
  options.mutable_output_tensor_uint_range()->set_max(255);
  EXPECT_THAT(Convert<uint8_t>(Tensor::ElementType::kUInt8, options),
              IsOkAndHolds(Expected<uint8_t>()));
}

TEST_P(ImageToTensorCalculatorMetalTest, Int8Output) {
  ImageToTensorCalculatorOptions options;
  options.mutable_output_tensor_int_range()->set_min(-128);
  options.mutable_output_tensor_int_range()->set_max(127);
  EXPECT_THAT(Convert<int8_t>(Tensor::ElementType::kInt8, options),
              IsOkAndHolds(Expected<int8_t>(/*offset=*/-128)));
}

// Returns a float16 RGBA image whose pixels cycle through R, G, B values
// below, inside and above [0, 1]: -1, 0.5 and 2.
absl::StatusOr<GpuBuffer> CreateOutOfRangeImage(int width, int height) {
  // IEEE 754 half-precision encodings of -1, 0.5, 2 and 1 (for alpha).
  constexpr uint16_t kValues[] = {0xBC00, 0x3800, 0x4000};
  constexpr uint16_t kOne = 0x3C00;
  ABSL_ASSIGN_OR_RETURN(CFHolder<CVPixelBufferRef> pixel_buffer,
                        CreateCVPixelBufferWithoutPool(
                            width, height, kCVPixelFormatType_64RGBAHalf));
  RET_CHECK_EQ(CVPixelBufferLockBaseAddress(*pixel_buffer, 0),
               kCVReturnSuccess);
  auto* data =
      static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(*pixel_buffer));
  RET_CHECK(data != nullptr);
  const size_t bytes_per_row = CVPixelBufferGetBytesPerRow(*pixel_buffer);
  std::vector<uint16_t> row(4 * width);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        row[4 * x + c] = kValues[(x + y + c) % 3];
      }
      row[4 * x + 3] = kOne;
    }
    std::memcpy(data + y * bytes_per_row, row.data(),
                row.size() * sizeof(uint16_t));
  }
  RET_CHECK_EQ(CVPixelBufferUnlockBaseAddress(*pixel_buffer, 0),
               kCVReturnSuccess);
  return GpuBuffer(pixel_buffer);
}

// Returns the expected 8-bit output for CreateOutOfRangeImage: values
// below / above [0, 1] are clamped to `min` / `max`, 0.5 maps to `mid`.
template <typename T>
std::vector<T> ExpectedClamped(const TestParams& params, T min, T mid, T max) {
  const T expected_values[] = {min, mid, max};
  std::vector<T> expected;
  for (int y = 0; y < params.height; ++y) {
    for (int x = 0; x < params.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        expected.push_back(expected_values[(x + y + c) % 3]);
      }
      if (params.num_channels == 4) {
        expected.push_back(0);
      }
    }
  }
  return expected;
}

TEST_P(ImageToTensorCalculatorMetalTest, UInt8OutputIsClamped) {
  const TestParams& params = GetParam();
  ImageToTensorCalculatorOptions options;
  options.mutable_output_tensor_uint_range()->set_min(0);
  options.mutable_output_tensor_uint_range()->set_max(255);
  // 0.5 * 255 = 127.5 is rounded to 128.
  MP_ASSERT_OK_AND_ASSIGN(GpuBuffer image,
                          CreateOutOfRangeImage(params.width, params.height));
  EXPECT_THAT(Convert<uint8_t>(Tensor::ElementType::kUInt8, options, image),
              IsOkAndHolds(ExpectedClamped<uint8_t>(params, 0, 128, 255)));
}

TEST_P(ImageToTensorCalculatorMetalTest, Int8OutputIsClamped) {
  const TestParams& params = GetParam();
  ImageToTensorCalculatorOptions options;
  options.mutable_output_tensor_int_range()->set_min(-128);
  options.mutable_output_tensor_int_range()->set_max(127);
  // 0.5 * 255 - 128 = -0.5 is rounded to 0.
  MP_ASSERT_OK_AND_ASSIGN(GpuBuffer image,
                          CreateOutOfRangeImage(params.width, params.height));
  EXPECT_THAT(Convert<int8_t>(Tensor::ElementType::kInt8, options, image),
              IsOkAndHolds(ExpectedClamped<int8_t>(params, -128, 0, 127)));
}

template <typename T>
std::vector<T> ExpectedShifted(
    const TestParams& params, const ImageFrame& frame,
    ImageToTensorCalculatorOptions::BorderMode border_mode, int offset) {
  std::vector<T> expected;
  for (int y = 0; y < params.height; ++y) {
    const uint8_t* row = frame.PixelData() + y * frame.WidthStep();
    for (int x = 0; x < params.width; ++x) {
      const int src_x = std::max(x - 1, 0);
      const bool zero =
          x == 0 && border_mode == ImageToTensorCalculatorOptions::BORDER_ZERO;
      for (int c = 0; c < 3; ++c) {
        const int value = zero ? 0 : row[4 * src_x + c];
        expected.push_back(static_cast<T>(value + offset));
      }
      if (params.num_channels == 4) {
        expected.push_back(0);
      }
    }
  }
  return expected;
}

TEST_P(ImageToTensorCalculatorMetalTest, ShiftedRoiUsesBorderMode) {
  const TestParams& params = GetParam();
  ImageFrame frame = CreateTestRgba8ImageFrame(params.width, params.height);
  for (const auto border_mode :
       {ImageToTensorCalculatorOptions::BORDER_ZERO,
        ImageToTensorCalculatorOptions::BORDER_REPLICATE}) {
    SCOPED_TRACE(border_mode);
    ImageToTensorCalculatorOptions options;
    options.mutable_output_tensor_uint_range()->set_min(0);
    options.mutable_output_tensor_uint_range()->set_max(255);
    options.set_border_mode(border_mode);
    // Output column x samples input column x - 1, so column 0 is outside the
    // image: 0 with BORDER_ZERO, input column 0 with BORDER_REPLICATE.
    EXPECT_THAT(
        Convert<uint8_t>(Tensor::ElementType::kUInt8, options,
                         /*image=*/std::nullopt,
                         /*roi_x_shift_pixels=*/-1.0f),
        IsOkAndHolds(ExpectedShifted<uint8_t>(params, frame, border_mode, 0)));
  }

  // With int8 output (beta == -128), out-of-bounds BORDER_ZERO pixels map to
  // -128 rather than 0.
  ImageToTensorCalculatorOptions int8_options;
  int8_options.mutable_output_tensor_int_range()->set_min(-128);
  int8_options.mutable_output_tensor_int_range()->set_max(127);
  int8_options.set_border_mode(ImageToTensorCalculatorOptions::BORDER_ZERO);
  EXPECT_THAT(
      Convert<int8_t>(Tensor::ElementType::kInt8, int8_options,
                      /*image=*/std::nullopt,
                      /*roi_x_shift_pixels=*/-1.0f),
      IsOkAndHolds(ExpectedShifted<int8_t>(
          params, frame, ImageToTensorCalculatorOptions::BORDER_ZERO, -128)));
}

INSTANTIATE_TEST_SUITE_P(
    ImageToTensorCalculatorMetalTests, ImageToTensorCalculatorMetalTest,
    testing::ValuesIn<TestParams>({
        // On devices, 4-channel float32 rows (16 bytes per pixel) always meet
        // the linear texture alignment, so the converter renders directly into
        // the output buffer. 4-channel 8-bit rows (4 bytes per pixel) only do
        // if the width is a multiple of 4 (with a 16-byte alignment): 8x2
        // renders directly, while 6x4 and 5x3 use the blit fallback.
        {"Rgba8x2", /*num_channels=*/4, /*width=*/8, /*height=*/2},
        {"Rgba6x4", /*num_channels=*/4, /*width=*/6, /*height=*/4},
        {"Rgba5x3", /*num_channels=*/4, /*width=*/5, /*height=*/3},
        {"Rgb8x2", /*num_channels=*/3, /*width=*/8, /*height=*/2},
        {"Rgb6x4", /*num_channels=*/3, /*width=*/6, /*height=*/4},
        // 15 pixels, so 3-channel 8-bit outputs have 45 bytes.
        {"Rgb5x3", /*num_channels=*/3, /*width=*/5, /*height=*/3},
    }),
    [](const testing::TestParamInfo<TestParams>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_METAL_ENABLED
