#include "mediapipe/gpu/gpu_test_base.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

#ifdef MEDIAPIPE_TENSOR_USE_AHWB
#include <android/hardware_buffer.h>

#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {

#if !MEDIAPIPE_DISABLE_GPU
class TensorAhwbTest : public mediapipe::GpuTestBase {
 public:
};

TEST_F(TensorAhwbTest, TestCpuThenAHWB) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
  {
    auto ptr = tensor.GetCpuWriteView().buffer<float>();
    EXPECT_NE(ptr, nullptr);
  }
  {
    auto ahwb = tensor.GetAHardwareBufferReadView().handle();
    EXPECT_NE(ahwb, nullptr);
  }
}

TEST_F(TensorAhwbTest, TestAHWBThenCpu) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
  {
    auto ahwb = tensor.GetAHardwareBufferWriteView().handle();
    EXPECT_NE(ahwb, nullptr);
  }
  {
    auto ptr = tensor.GetCpuReadView().buffer<float>();
    EXPECT_NE(ptr, nullptr);
  }
}

TEST_F(TensorAhwbTest, TestCpuThenGl) {
  RunInGlContext([] {
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    {
      auto ptr = tensor.GetCpuWriteView().buffer<float>();
      EXPECT_NE(ptr, nullptr);
    }
    {
      auto ssbo = tensor.GetOpenGlBufferReadView().name();
      EXPECT_GT(ssbo, 0);
    }
  });
}

}  // namespace mediapipe

#endif  // !MEDIAPIPE_DISABLE_GPU

#endif  // MEDIAPIPE_TENSOR_USE_AHWB
