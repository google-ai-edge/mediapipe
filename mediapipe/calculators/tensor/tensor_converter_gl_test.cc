#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "mediapipe/calculators/tensor/tensor_converter_gl30.h"
#include "mediapipe/calculators/tensor/tensor_converter_gl31.h"
#include "mediapipe/calculators/tensor/tensor_converter_gpu.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_test_base.h"
#include "mediapipe/util/image_test_utils.h"

namespace mediapipe {
namespace {

constexpr std::pair<float, float> kDefaultOutputRange =
    std::pair<float, float>({0.0f, 1.0f});

constexpr float kEpsilon = 1e-4f;

enum GlVersion { GlVersion30, GlVersion31 };

class TensorConverterGlTest : public GpuTestWithParamBase<GlVersion> {
  void SetUp() override { GpuTestWithParamBase::SetUp(); }

 protected:
  absl::StatusOr<std::unique_ptr<TensorConverterGpu>> CreateTensorConverter(
      int width, int height, const std::pair<float, float>& output_range,
      bool include_alpha, bool single_channel, bool flip_vertically,
      int num_output_channels) {
    std::unique_ptr<TensorConverterGpu> tensor_converter;
    switch (GetParam()) {
      case GlVersion30:
        return CreateTensorConverterGl30(helper_, &memory_manager_, width,
                                         height, output_range, include_alpha,
                                         single_channel, flip_vertically,
                                         num_output_channels);
        break;
      case GlVersion31:
        return CreateTensorConverterGl31(helper_, &memory_manager_, width,
                                         height, output_range, include_alpha,
                                         single_channel, flip_vertically,
                                         num_output_channels);
        break;
    }
    return absl::InternalError("Unknown GlVersion: " +
                               std::to_string(GetParam()));
  }

  MemoryManager memory_manager_;
};

INSTANTIATE_TEST_SUITE_P(ConfigValues, TensorConverterGlTest,
                         testing::Values(GlVersion::GlVersion30,
                                         GlVersion::GlVersion31));

TEST_P(TensorConverterGlTest, ConvertFloat32ImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    GpuBuffer input = CreateTestFloat32GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/false, /*single_channel=*/true,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/1));

    Tensor output = tensor_converter->Convert(input);

    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int i = 0; i < input.width() * input.height(); ++i) {
      EXPECT_NEAR(tensor_ptr[i],
                  reinterpret_cast<const float*>(input_view->PixelData())[i],
                  kEpsilon);
    }
  });
}

TEST_P(TensorConverterGlTest, ConvertScaledFloat32ImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    GpuBuffer input = CreateTestFloat32GpuBuffer(/*width=*/3, /*height=*/4);
    const std::pair<float, float> output_range = {-1.0f, 1.0f};
    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(), output_range,
                              /*include_alpha=*/false, /*single_channel=*/true,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/1));

    Tensor output = tensor_converter->Convert(input);

    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int i = 0; i < input.width() * input.height(); ++i) {
      EXPECT_NEAR(tensor_ptr[i],
                  reinterpret_cast<const float*>(input_view->PixelData())[i] *
                          (output_range.second - output_range.first) +
                      output_range.first,
                  kEpsilon);
    }
  });
}

TEST_P(TensorConverterGlTest, ConvertGrey8ImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    GpuBuffer input = CreateTestGrey8GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/false, /*single_channel=*/true,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/1));

    Tensor output = tensor_converter->Convert(input);
    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int i = 0; i < input.width() * input.height(); ++i) {
      EXPECT_NEAR(tensor_ptr[i],
                  static_cast<float>(input_view->PixelData()[i]) / 255.0f,
                  kEpsilon);
    }
  });
}

TEST_P(TensorConverterGlTest, ConvertRgbaImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    constexpr int kNumChannels = 4;
    GpuBuffer input = CreateTestRgba8GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/true, /*single_channel=*/false,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/4));

    Tensor output = tensor_converter->Convert(input);
    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int i = 0; i < input.width() * input.height() * kNumChannels; ++i) {
      EXPECT_NEAR(tensor_ptr[i],
                  static_cast<float>(input_view->PixelData()[i]) / 255.0,
                  kEpsilon);
    }
  });
}

TEST_P(TensorConverterGlTest,
       ConvertRgbaImageFrameExcludingAlphaToTensorOnGpu) {
  RunInGlContext([this]() {
    constexpr int kNumInputChannel = 4;
    constexpr int kNumOutputChannel = 3;
    GpuBuffer input = CreateTestRgba8GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/false, /*single_channel=*/false,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/3));

    Tensor output = tensor_converter->Convert(input);
    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int i = 0; i < input.width() * input.height(); ++i) {
      for (int channel = 0; channel < kNumOutputChannel; ++channel) {
        EXPECT_NEAR(
            tensor_ptr[i * kNumOutputChannel + channel],
            static_cast<float>(
                input_view->PixelData()[i * kNumInputChannel + channel]) /
                255.0,
            kEpsilon);
      }
    }
  });
}

TEST_P(TensorConverterGlTest, ConvertFlippedFloat32ImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    GpuBuffer input = CreateTestFloat32GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/false, /*single_channel=*/true,
                              /*flip_vertically=*/true,
                              /*num_output_channels=*/1));

    Tensor output = tensor_converter->Convert(input);

    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    const int num_pixels = input.width() * input.height();
    for (int i = 0; i < num_pixels; ++i) {
      const int x = i % input.width();
      const int y = i / input.width();
      const int flipped_y = input.height() - y - 1;
      const int index = flipped_y * input.width() + x;
      EXPECT_NEAR(tensor_ptr[index],
                  reinterpret_cast<const float*>(input_view->PixelData())[i],
                  kEpsilon);
    }
  });
}

TEST_P(TensorConverterGlTest, ConvertFlippedRgbaImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    constexpr int kNumChannels = 4;
    GpuBuffer input = CreateTestRgba8GpuBuffer(/*width=*/3, /*height=*/2);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/true, /*single_channel=*/false,
                              /*flip_vertically=*/true,
                              /*num_output_channels=*/kNumChannels));

    Tensor output = tensor_converter->Convert(input);

    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    for (int y = 0; y < input.height(); ++y) {
      auto* flipped_row_ptr =
          input_view->PixelData() +
          input.width() * (input.height() - y - 1) * kNumChannels;
      for (int x = 0; x < input.width(); ++x) {
        auto* pixel_ptr = flipped_row_ptr + x * kNumChannels;
        for (int channel = 0; channel < kNumChannels; ++channel) {
          EXPECT_NEAR(tensor_ptr[y * input.width() * kNumChannels +
                                 x * kNumChannels + channel],
                      *(pixel_ptr + channel) / 255.0f, kEpsilon);
        }
      }
    }
  });
}

TEST_P(TensorConverterGlTest,
       ConvertSingleChannelOfRgbaImageFrameToTensorOnGpu) {
  RunInGlContext([this]() {
    GpuBuffer input = CreateTestRgba8GpuBuffer(/*width=*/3, /*height=*/4);

    MP_ASSERT_OK_AND_ASSIGN(
        auto tensor_converter,
        CreateTensorConverter(input.width(), input.height(),
                              kDefaultOutputRange,
                              /*include_alpha=*/false, /*single_channel=*/true,
                              /*flip_vertically=*/false,
                              /*num_output_channels=*/1));

    Tensor output = tensor_converter->Convert(input);

    const auto input_view = input.GetReadView<ImageFrame>();
    const auto cpu_read_view = output.GetCpuReadView();
    const float* tensor_ptr = cpu_read_view.buffer<float>();
    const int num_pixels = input.width() * input.height();
    const int num_channels = input_view->NumberOfChannels();
    constexpr int kSelectedChannel = 0;
    for (int i = 0; i < num_pixels; ++i) {
      EXPECT_NEAR(
          tensor_ptr[i],
          static_cast<float>(
              input_view->PixelData()[i * num_channels + kSelectedChannel]) /
              255.0,
          kEpsilon);
    }
  });
}

}  // anonymous namespace
}  // namespace mediapipe
