#include "mediapipe/framework/formats/tensor.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {

TEST(TensorAhwbTest, TestCpuThenAHWB) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
  {
    auto ptr = tensor.GetCpuWriteView().buffer<float>();
    EXPECT_NE(ptr, nullptr);
  }
  {
    auto view = tensor.GetAHardwareBufferReadView();
    EXPECT_NE(view.handle(), nullptr);
    view.SetReadingFinishedFunc([](bool) { return true; });
  }
}

TEST(TensorAhwbTest, TestAHWBThenCpu) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
  {
    auto view = tensor.GetAHardwareBufferWriteView();
    EXPECT_NE(view.handle(), nullptr);
    view.SetWritingFinishedFD(-1, [](bool) { return true; });
  }
  {
    auto ptr = tensor.GetCpuReadView().buffer<float>();
    EXPECT_NE(ptr, nullptr);
  }
}

}  // namespace mediapipe
