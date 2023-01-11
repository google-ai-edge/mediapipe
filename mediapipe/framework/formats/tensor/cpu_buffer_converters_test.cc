#include <cstdint>

#include "mediapipe/framework/formats/tensor/tensor2.h"
#include "mediapipe/framework/formats/tensor/views/buffer.h"
#include "mediapipe/framework/formats/tensor/views/cpu_buffer.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

MATCHER_P(NearWithPrecision, precision, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < precision;
}
MATCHER_P(IntegerEqual, precision, "") {
  return std::get<0>(arg) == std::get<1>(arg);
}

namespace mediapipe {

TEST(TensorCpuViewTest, TestWrite32ThenRead16) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<float>() = 1234.0f;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat16}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<Float16>(), 1234.0f);
  }
}

TEST(TensorCpuViewTest, TestWrite16ThenRead32) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat16}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<Float16>() = 1234.0f;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<float>(), 1234.0f);
  }
}

TEST(TensorCpuViewTest, TestWriteFloat32ThenReadInt8) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<float>() = 0.121569f;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(
        *view->data<uint8_t>(),
        static_cast<uint8_t>(0.121569f * std::numeric_limits<uint8_t>::max()));
  }
}

TEST(TensorCpuViewTest, TestWriteInt8ThenReadFloat32) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<uint8_t>() =
        static_cast<uint8_t>(0.123f * std::numeric_limits<uint8_t>::max());
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_NEAR(*view->data<float>(), 0.123f,
                1.0f / std::numeric_limits<uint8_t>::max());
  }
}

TEST(TensorCpuViewTest, TestWriteUInt8ThenReadUInt16) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<uint8_t>() = 123;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kUInt16}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<uint16_t>(), uint16_t{123} << 8);
  }
}

TEST(TensorCpuViewTest, TestWriteUInt16ThenReadUInt8) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kUInt16}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<uint16_t>() = uint16_t{123} << 8;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<uint8_t>(), 123);
  }
}

TEST(TensorCpuViewTest, TestWriteNegativeInt8ThenReadUInt8) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<int8_t>() = -123;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<uint8_t>(), 0);
  }
}

TEST(TensorCpuViewTest, TestWritePositiveInt8ThenReadUInt8) {
  Tensor2 tensor{Tensor2::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    *view->data<int8_t>() = 123;
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kUInt8}}));
    ASSERT_NE(view->data<void>(), nullptr);
    EXPECT_EQ(*view->data<uint8_t>(), 123 * 2);
  }
}

TEST(TensorCpuViewTest, TestDequantization) {
  constexpr int num_elements = 20;
  // Gives quantization values in range [-100, 90].
  constexpr int zero_point = -100;
  constexpr float scale = 2.0f;
  Tensor2 tensor{Tensor2::Shape({num_elements})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {
                    .format = TensorBufferDescriptor::Format::kQuantizedInt8,
                    .quantization_parameters = {.scale = scale,
                                                .zero_point = zero_point}}}));
    ASSERT_NE(view->data<void>(), nullptr);
    auto data = view->data<int8_t>();
    for (int i = 0; i < num_elements; ++i) {
      // Add some bias (+1) to make round-up take place.
      data[i] = (i * 20 + 1) / scale + zero_point;
    }
  }
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    std::vector<float> reference(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      reference[i] = i * 20.0f + 1.0f;
    }
    EXPECT_THAT(absl::Span<float>(view->data<float>(), num_elements),
                testing::Pointwise(NearWithPrecision(1.001), reference));
  }
}

TEST(TensorCpuViewTest, TestQuantization) {
  constexpr int num_elements = 20;
  // Gives quantization values in range [-100, 90].
  constexpr int zero_point = -100;
  constexpr float scale = 2.0f;
  Tensor2 tensor{Tensor2::Shape({num_elements})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor2::View::Access::kWriteOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    ASSERT_NE(view->data<void>(), nullptr);
    auto data = view->data<float>();
    for (int i = 0; i < num_elements; ++i) {
      // Add some bias (+1) to make round-up take place.
      data[i] = i * 20 + 1;
    }
  }
  {
    TensorCpuViewDescriptor d{
        .buffer = {.format = TensorBufferDescriptor::Format::kQuantizedInt8,
                   .quantization_parameters = {.scale = scale,
                                               .zero_point = zero_point}}};
    MP_ASSERT_OK_AND_ASSIGN(
        auto view, tensor.GetView<Tensor2::View::Access::kReadOnly>(d));
    ASSERT_NE(view->data<void>(), nullptr);
    std::vector<int8_t> reference(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      reference[i] = (i * 20 + 1) / scale + zero_point;
    }
    EXPECT_THAT(absl::Span<int8_t>(view->data<int8_t>(), num_elements),
                testing::Pointwise(IntegerEqual(0), reference));
  }
}

}  // namespace mediapipe
