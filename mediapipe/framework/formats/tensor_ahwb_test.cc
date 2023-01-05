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

TEST(TensorAhwbTest, TestAhwbAlignment) {
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{5});
  {
    auto view = tensor.GetAHardwareBufferWriteView(16);
    ASSERT_NE(view.handle(), nullptr);
    if (__builtin_available(android 26, *)) {
      AHardwareBuffer_Desc desc;
      AHardwareBuffer_describe(view.handle(), &desc);
      // sizeof(float) * 5 = 20, the closest aligned to 16 size is 32.
      EXPECT_EQ(desc.width, 32);
    }
    view.SetWritingFinishedFD(-1, [](bool) { return true; });
  }
}

// Tensor::GetCpuView uses source location mechanism that gives source file name
// and line from where the method is called. The function is intended just to
// have two calls providing the same source file name and line.
auto GetCpuView(const Tensor &tensor) { return tensor.GetCpuWriteView(); }

// The test checks the tracking mechanism: when a tensor's Cpu view is retrieved
// for the first time then the source location is attached to the tensor. If the
// Ahwb view is requested then from the tensor then the previously recorded Cpu
// view request source location is marked for using Ahwb storage.
// When a Cpu view with the same source location (but for the newly allocated
// tensor) is requested and the location is marked to use Ahwb storage then the
// Ahwb storage is allocated for the CpuView.
TEST(TensorAhwbTest, TestTrackingAhwb) {
  // Create first tensor and request Cpu and then Ahwb view to mark the source
  // location for Ahwb storage.
  {
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{9});
    {
      auto view = GetCpuView(tensor);
      EXPECT_NE(view.buffer<float>(), nullptr);
    }
    {
      // Align size of the Ahwb by multiple of 16.
      auto view = tensor.GetAHardwareBufferWriteView(16);
      EXPECT_NE(view.handle(), nullptr);
      view.SetReadingFinishedFunc([](bool) { return true; });
    }
  }
  {
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{9});
    {
      // The second tensor uses the same Cpu view source location so Ahwb
      // storage is allocated internally.
      auto view = GetCpuView(tensor);
      EXPECT_NE(view.buffer<float>(), nullptr);
    }
    {
      // Check the Ahwb size to be aligned to multiple of 16. The alignment is
      // stored by previous requesting of the Ahwb view.
      auto view = tensor.GetAHardwareBufferReadView();
      EXPECT_NE(view.handle(), nullptr);
      if (__builtin_available(android 26, *)) {
        AHardwareBuffer_Desc desc;
        AHardwareBuffer_describe(view.handle(), &desc);
        // sizeof(float) * 9 = 36. The closest aligned size is 48.
        EXPECT_EQ(desc.width, 48);
      }
      view.SetReadingFinishedFunc([](bool) { return true; });
    }
  }
}

}  // namespace mediapipe
