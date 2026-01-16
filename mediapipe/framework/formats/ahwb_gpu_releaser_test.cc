
#include "mediapipe/framework/port.h"  // IWYU pragma: keep

#if MEDIAPIPE_TENSOR_USE_AHWB

#include <memory>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/gtest.h"

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

namespace mediapipe {
namespace {

class ReleaseTracker {};

TEST(AhwbGpuReleaserTest, ShouldImmediatelyReleaseForAhwbOnlyUsage) {
  std::shared_ptr<ReleaseTracker> to_be_released =
      std::make_shared<ReleaseTracker>();
  std::weak_ptr<ReleaseTracker> weak_to_be_released = to_be_released;

  {
    Tensor tensor({Tensor::ElementType::kFloat32, Tensor::Shape({123})});
    {
      // Request Ahwb first to get Ahwb storage allocated internally.
      auto view = tensor.GetAHardwareBufferWriteView();
      ASSERT_NE(view.handle(), nullptr);
      view.SetWritingFinishedFD(
          -1,
          [to_be_released = std::move(to_be_released)](bool) { return true; });
    }
    // Destruction of the tensor will trigger the immediate buffer release.
    EXPECT_FALSE(weak_to_be_released.expired());
  }
  EXPECT_TRUE(weak_to_be_released.expired());
}

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
TEST(AhwbGpuReleaserTest,
     ShouldDelayReleaseAhwbGpuUsageDuringGlcontextDestruction) {
  struct GpuResourcesHolder {
    GpuSharedData gpu_shared;
    std::shared_ptr<GpuResources> gpu_resources = gpu_shared.gpu_resources;
    std::shared_ptr<GlContext> gl_context = gpu_resources->gl_context();
  };
  std::unique_ptr<GpuResourcesHolder> gpu_resources_holder =
      std::make_unique<GpuResourcesHolder>();

  std::shared_ptr<ReleaseTracker> to_be_released =
      std::make_shared<ReleaseTracker>();
  std::weak_ptr<ReleaseTracker> weak_to_be_released = to_be_released;
  bool can_release = false;

  std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>(
      Tensor::ElementType::kFloat32, Tensor::Shape({123}));
  {
    // Request Ahwb first to get Ahwb storage allocated internally.
    auto view = tensor->GetAHardwareBufferWriteView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetWritingFinishedFD(
        -1, [&, to_be_released = std::move(to_be_released)](bool) {
          return can_release;
        });
  }
  // Destruction of the tensor will trigger the release to the delayed
  // releaser.
  EXPECT_FALSE(weak_to_be_released.expired());

  // GPU usage requires to respect the writing finish signal.
  gpu_resources_holder->gl_context->Run([&] {
    auto ssbo_view = tensor->GetOpenGlBufferWriteView();
    auto ssbo_name = ssbo_view.name();
    ASSERT_GT(ssbo_name, 0);
  });

  tensor.reset();
  // Buffer is not released yet event though the tensor is destroyed.
  EXPECT_FALSE(weak_to_be_released.expired());
  // Now we can allow the release.
  can_release = true;
  // Destruction of the gpu resources will trigger the release of the buffer.
  gpu_resources_holder.reset();
  // Buffer is now released.
  EXPECT_TRUE(weak_to_be_released.expired());
}

TEST(AhwbGpuReleaserTest,
     ShouldDelayReleaseAhwbGpuUsageForSubsequentTensorRelease) {
  struct GpuResourcesHolder {
    GpuSharedData gpu_shared;
    std::shared_ptr<GpuResources> gpu_resources = gpu_shared.gpu_resources;
    std::shared_ptr<GlContext> gl_context = gpu_resources->gl_context();
  };
  std::unique_ptr<GpuResourcesHolder> gpu_resources_holder =
      std::make_unique<GpuResourcesHolder>();

  std::shared_ptr<ReleaseTracker> to_be_released =
      std::make_shared<ReleaseTracker>();
  std::weak_ptr<ReleaseTracker> weak_to_be_released = to_be_released;
  bool can_release = false;

  std::unique_ptr<Tensor> tensor = std::make_unique<Tensor>(
      Tensor::ElementType::kFloat32, Tensor::Shape({123}));
  {
    // Request Ahwb first to get Ahwb storage allocated internally.
    auto view = tensor->GetAHardwareBufferWriteView();
    ASSERT_NE(view.handle(), nullptr);
    view.SetWritingFinishedFD(
        -1, [&, to_be_released = std::move(to_be_released)](bool) {
          return can_release;
        });
  }
  EXPECT_FALSE(weak_to_be_released.expired());

  // GPU usage requires to respect the writing finish signal.
  gpu_resources_holder->gl_context->Run([&] {
    auto ssbo_view = tensor->GetOpenGlBufferWriteView();
    auto ssbo_name = ssbo_view.name();
    ASSERT_GT(ssbo_name, 0);
  });

  tensor.reset();
  // Buffer is not released yet event though the tensor is destroyed.
  EXPECT_FALSE(weak_to_be_released.expired());
  // Now we can allow the release.
  can_release = true;

  {
    bool tensor2_second_release_attempt = false;
    // Create a new tensor to trigger another buffer release.
    Tensor tensor2({Tensor::ElementType::kFloat32, Tensor::Shape({123})});
    {
      auto view = tensor2.GetAHardwareBufferWriteView();
      ASSERT_NE(view.handle(), nullptr);
      view.SetWritingFinishedFD(-1, [&](bool) {
        bool release_now = tensor2_second_release_attempt;
        // Release on second attempt. This way, the second buffer will be first
        // pushed to the AhwbGpuReleaser. This step trigger the releases of the
        // first buffer which now can be released.
        tensor2_second_release_attempt = true;
        return release_now;
      });
    }
    gpu_resources_holder->gl_context->Run([&] {
      auto ssbo_view = tensor2.GetOpenGlBufferWriteView();
      auto ssbo_name = ssbo_view.name();
      ASSERT_GT(ssbo_name, 0);
    });
  }

  EXPECT_TRUE(weak_to_be_released.expired());
}
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

}  // namespace
}  // namespace mediapipe

#endif  // MEDIAPIPE_TENSOR_USE_AHWB
