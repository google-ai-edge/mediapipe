
#if !defined(MEDIAPIPE_NO_JNI) && \
    (__ANDROID_API__ >= 26 ||     \
     defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
#include <android/hardware_buffer.h>

#include <cstdint>

#include "mediapipe/framework/formats/tensor_cpu_buffer.h"
#include "mediapipe/framework/formats/tensor_hardware_buffer.h"
#include "mediapipe/framework/formats/tensor_v2.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {

namespace {

class TensorHardwareBufferTest : public ::testing::Test {
 public:
  TensorHardwareBufferTest() {}
  ~TensorHardwareBufferTest() override {}
};

TEST_F(TensorHardwareBufferTest, TestFloat32) {
  Tensor tensor{Tensor::Shape({1})};
  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor::View::Access::kWriteOnly>(
            TensorHardwareBufferViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    EXPECT_NE(view->handle(), nullptr);
  }
  {
    const auto& const_tensor = tensor;
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        const_tensor.GetView<Tensor::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format =
                               TensorBufferDescriptor::Format::kFloat32}}));
    EXPECT_NE(view->data<void>(), nullptr);
  }
}

TEST_F(TensorHardwareBufferTest, TestInt8Padding) {
  Tensor tensor{Tensor::Shape({1})};

  {
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        tensor.GetView<Tensor::View::Access::kWriteOnly>(
            TensorHardwareBufferViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kInt8,
                           .size_alignment = 4}}));
    EXPECT_NE(view->handle(), nullptr);
  }
  {
    const auto& const_tensor = tensor;
    MP_ASSERT_OK_AND_ASSIGN(
        auto view,
        const_tensor.GetView<Tensor::View::Access::kReadOnly>(
            TensorCpuViewDescriptor{
                .buffer = {.format = TensorBufferDescriptor::Format::kInt8}}));
    EXPECT_NE(view->data<void>(), nullptr);
  }
}

}  // namespace

}  // namespace mediapipe

#endif  // !defined(MEDIAPIPE_NO_JNI) && (__ANDROID_API__ >= 26 ||
        // defined(__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__))
