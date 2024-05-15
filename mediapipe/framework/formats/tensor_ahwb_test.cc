#include <android/hardware_buffer.h>

#include <array>
#include <memory>

#include "mediapipe/framework/formats/hardware_buffer.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/memory_manager.h"
#include "mediapipe/gpu/multi_pool.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mediapipe {
namespace {

using ::testing::Each;
using ::testing::Return;
using ::testing::Truly;

MultiPoolOptions GetTestMultiPoolOptions() {
  MultiPoolOptions options;
  options.min_requests_before_pool = 0;
  return options;
}

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

TEST(TensorAhwbTest, EveryAhwbReadViewReleaseCallbackIsInvoked) {
  constexpr int kNumReleaseCallbacks = 10;
  std::array<bool, kNumReleaseCallbacks> callbacks_invoked;
  callbacks_invoked.fill(false);

  {
    // Create tensor.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    {
      auto ptr = tensor.GetCpuWriteView().buffer<float>();
      EXPECT_NE(ptr, nullptr);
    }

    // Get AHWB read view multiple times (e.g. simulating how multiple inference
    // calculators could read from the same tensor)
    for (int i = 0; i < kNumReleaseCallbacks; ++i) {
      auto view = tensor.GetAHardwareBufferReadView();
      EXPECT_NE(view.handle(), nullptr);
      view.SetReleaseCallback(
          [&callbacks_invoked, i] { callbacks_invoked[i] = true; });
    }

    // Destroy tensor on scope exit triggering release callbacks.
  }

  EXPECT_THAT(callbacks_invoked, Each(true));
}

TEST(TensorAhwbTest,
     GetAHardwareBufferReadViewTriggersReleaseForFinishedReads) {
  constexpr int kNumReleaseCallbacks = 10;
  std::array<bool, kNumReleaseCallbacks> release_callbacks_invoked;
  release_callbacks_invoked.fill(false);

  {
    // Create tensor.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    {
      auto ptr = tensor.GetCpuWriteView().buffer<float>();
      ASSERT_NE(ptr, nullptr);
    }

    // Get AHWB read view multiple times (e.g. simulating how multiple inference
    // calculators could read from the same tensor)
    for (int i = 0; i < kNumReleaseCallbacks; ++i) {
      if (i > 0) {
        ASSERT_FALSE(release_callbacks_invoked[i - 1]);
      }
      // Triggers cleanup for a previous ready read.
      auto view = tensor.GetAHardwareBufferReadView();
      ASSERT_NE(view.handle(), nullptr);
      if (i > 0) {
        // Triggered cleanup for a previous read as it's ready.
        ASSERT_TRUE(release_callbacks_invoked[i - 1]);
      }

      // Marking as a finished read.
      view.SetReadingFinishedFunc([](bool) { return true; });
      view.SetReleaseCallback([&release_callbacks_invoked, i] {
        release_callbacks_invoked[i] = true;
      });
    }
    ASSERT_FALSE(release_callbacks_invoked[kNumReleaseCallbacks - 1]);

    // Destroy tensor on scope exit triggering last release callback.
  }

  EXPECT_THAT(release_callbacks_invoked, Each(true));
}

TEST(TensorAhwbTest, GetAhwbReadViewDoesNotTriggerReleaseForUnfinishedReads) {
  constexpr int kNumReleaseCallbacks = 10;
  std::array<bool, kNumReleaseCallbacks> release_callbacks_invoked;
  release_callbacks_invoked.fill(false);

  {
    // Create tensor.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    {
      auto ptr = tensor.GetCpuWriteView().buffer<float>();
      ASSERT_NE(ptr, nullptr);
    }

    // Get AHWB read view multiple times (e.g. simulating how multiple inference
    // calculators could read from the same tensor)
    bool is_reading_finished = false;
    for (int i = 0; i < kNumReleaseCallbacks; ++i) {
      auto view = tensor.GetAHardwareBufferReadView();
      ASSERT_NE(view.handle(), nullptr);

      // Marking as an unfinished read.
      view.SetReadingFinishedFunc(
          [&is_reading_finished](bool) { return is_reading_finished; });
      view.SetReleaseCallback([&release_callbacks_invoked, i] {
        release_callbacks_invoked[i] = true;
      });
    }
    ASSERT_THAT(release_callbacks_invoked, Each(false));

    // Destroy tensor on scope exit triggering release callbacks considering
    // reads are finished.
    is_reading_finished = true;
  }

  EXPECT_THAT(release_callbacks_invoked, Each(true));
}

TEST(TensorAhwbTest, EveryAhwbWriteViewReleaseCallbackIsInvoked) {
  constexpr int kNumReleaseCallbacks = 10;
  std::array<bool, kNumReleaseCallbacks> callbacks_invoked;
  callbacks_invoked.fill(false);

  {
    // Create tensor.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    // Get AHWB write view multiple times and set release callback.
    for (int i = 0; i < kNumReleaseCallbacks; ++i) {
      auto view = tensor.GetAHardwareBufferWriteView();
      EXPECT_NE(view.handle(), nullptr);
      view.SetReleaseCallback(
          [&callbacks_invoked, i] { callbacks_invoked[i] = true; });
    }

    // Destroy tensor on scope exit triggering release callbacks.
  }

  EXPECT_THAT(callbacks_invoked, Each(true));
}

TEST(TensorAhwbTest,
     EveryAhwbWriteViewReleaseCallbackIsInvokedWritingFininshedSpecified) {
  constexpr int kNumReleaseCallbacks = 10;
  std::array<bool, kNumReleaseCallbacks> release_callbacks_invoked;
  release_callbacks_invoked.fill(false);

  {
    // Create tensor.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{1});
    // Get AHWB write view multiple times and set release callback.
    for (int i = 0; i < kNumReleaseCallbacks; ++i) {
      if (i > 0) {
        ASSERT_FALSE(release_callbacks_invoked[i - 1]);
      }
      auto view = tensor.GetAHardwareBufferWriteView();
      if (i > 0) {
        ASSERT_TRUE(release_callbacks_invoked[i - 1]);
      }
      ASSERT_NE(view.handle(), nullptr);
      view.SetWritingFinishedFD(/*dummy fd=*/-1, [](bool) { return true; });
      view.SetReleaseCallback([&release_callbacks_invoked, i] {
        release_callbacks_invoked[i] = true;
      });
    }

    ASSERT_FALSE(release_callbacks_invoked[kNumReleaseCallbacks - 1]);

    // Destroy tensor on scope exit triggering last release callback.
  }

  EXPECT_THAT(release_callbacks_invoked, Each(true));
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
  Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{5},
                /*memory_manager=*/nullptr, /*memory_alignment=*/16);
  {
    auto view = tensor.GetAHardwareBufferWriteView();
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
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{9},
                  /*memory_manager=*/nullptr, /*memory_alignment=*/16);
    {
      auto view = GetCpuView(tensor);
      EXPECT_NE(view.buffer<float>(), nullptr);
    }
    {
      // Align size of the Ahwb by multiple of 16.
      auto view = tensor.GetAHardwareBufferWriteView();
      EXPECT_NE(view.handle(), nullptr);
      view.SetReadingFinishedFunc([](bool) { return true; });
    }
  }
  {
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{9},
                  /*memory_manager=*/nullptr, /*memory_alignment=*/16);
    {
      // The second tensor uses the same Cpu view source location so Ahwb
      // storage is allocated internally.
      auto view = GetCpuView(tensor);
      EXPECT_NE(view.buffer<float>(), nullptr);
      EXPECT_TRUE(tensor.ready_as_ahwb());
    }
  }
}

TEST(TensorAhwbTest, ShouldReuseHardwareBufferFromHardwareBufferPool) {
  constexpr int kTensorSize = 123;
  MemoryManager memory_manager(GetTestMultiPoolOptions());

  AHardwareBuffer *buffer = nullptr;
  {
    // First call instantiates HardwareBuffer.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kTensorSize},
                  &memory_manager);
    auto view = tensor.GetAHardwareBufferWriteView();
    buffer = view.handle();
    EXPECT_NE(buffer, nullptr);
  }
  {
    // Second request should return the same AHardwareBuffer (handle).
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kTensorSize},
                  &memory_manager);
    auto view = tensor.GetAHardwareBufferWriteView();
    EXPECT_EQ(view.handle(), buffer);
  }
}

TEST(TensorAhwbTest, ShouldNotReuseHardwareBufferFromHardwareBufferPool) {
  constexpr int kTensorASize = 123;
  constexpr int kTensorBSize = 456;
  MemoryManager memory_manager(GetTestMultiPoolOptions());

  AHardwareBuffer *buffer = nullptr;
  {
    // First call instantiates HardwareBuffer of size kTensorASize.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kTensorASize},
                  &memory_manager);
    auto view = tensor.GetAHardwareBufferWriteView();
    buffer = view.handle();
    EXPECT_NE(buffer, nullptr);
  }
  {
    // Second call creates a second HardwareBuffer of size kTensorBSize.
    Tensor tensor(Tensor::ElementType::kFloat32, Tensor::Shape{kTensorBSize},
                  &memory_manager);
    auto view = tensor.GetAHardwareBufferWriteView();
    EXPECT_NE(view.handle(), buffer);
  }
}

}  // namespace
}  // namespace mediapipe
