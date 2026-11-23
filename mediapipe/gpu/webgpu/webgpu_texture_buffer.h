#ifndef MEDIAPIPE_GPU_WEBGPU_WEBGPU_TEXTURE_BUFFER_H_
#define MEDIAPIPE_GPU_WEBGPU_WEBGPU_TEXTURE_BUFFER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"          // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_check.h"            // from @com_google_absl
#include "absl/status/statusor.h"           // from @com_google_absl
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/gpu/multi_pool.h"
#include "mediapipe/gpu/reusable_pool.h"
#include "mediapipe/gpu/webgpu/webgpu_headers.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"

namespace mediapipe {

class WebGpuTextureBuffer
    : public internal::GpuBufferStorageImpl<
          WebGpuTextureBuffer, internal::ViewProvider<WebGpuTextureView>> {
 public:
  static std::unique_ptr<WebGpuTextureBuffer> Create(
      const wgpu::Instance& instance, const wgpu::Device& device,
      uint32_t width, uint32_t height,
      GpuBufferFormat format = GpuBufferFormat::kRGBA32);

  static std::unique_ptr<WebGpuTextureBuffer> Create(uint32_t width,
                                                     uint32_t height,
                                                     GpuBufferFormat format);

  // Creates WebGpuTextureBuffer.
  //
  // NOTE: you can provide a custom texture deleter to wrap a `texture` and
  //   avoid its destruction (no-op deleter) or perform custom actions on
  //   deletion if needed.
  WebGpuTextureBuffer(
      wgpu::Instance instance, wgpu::Device device, wgpu::Texture texture,
      uint32_t width, uint32_t height, GpuBufferFormat format,
      /*absl_nonnull - not yet supported*/
      absl::AnyInvocable<void(wgpu::Texture*)> texture_deleter =
          [](wgpu::Texture* texture) { texture->Destroy(); })
      : instance_(std::move(instance)),
        device_(std::move(device)),
        texture_(std::move(texture)),
        width_(width),
        height_(height),
        format_(format),
        texture_deleter_(std::move(texture_deleter)) {
    ABSL_CHECK(texture_deleter_);
  }

  ~WebGpuTextureBuffer() override { texture_deleter_(&texture_); }

  int width() const override { return width_; }
  int height() const override { return height_; }

  WebGpuTextureView GetReadView(
      internal::types<WebGpuTextureView>) const override;

  WebGpuTextureView GetWriteView(internal::types<WebGpuTextureView>) override;

  GpuBufferFormat format() const override { return format_; }

  const wgpu::Device& device() const { return device_; }
  const wgpu::Instance& instance() const { return instance_; }

  // TODO: implement.
  void Reuse() {}

 private:
  wgpu::Instance instance_;
  wgpu::Device device_;
  wgpu::Texture texture_;
  const uint32_t width_ = 0;
  const uint32_t height_ = 0;
  const GpuBufferFormat format_;
  absl::AnyInvocable<void(wgpu::Texture*)> texture_deleter_;
};

class Canvas;
Canvas& RenderToWebGpuCanvas(const std::shared_ptr<WebGpuTextureBuffer>& input,
                             int canvas_width, int canvas_height);

class WebGpuTextureBufferPool : public ReusablePool<WebGpuTextureBuffer> {
 public:
  static std::shared_ptr<WebGpuTextureBufferPool> Create(
      const wgpu::Instance& instance, const wgpu::Device& device,
      const internal::GpuBufferSpec& spec, const MultiPoolOptions& options) {
    return std::make_shared<WebGpuTextureBufferPool>(instance, device, spec,
                                                     options);
  }

  static absl::StatusOr<std::shared_ptr<WebGpuTextureBuffer>>
  CreateBufferWithoutPool(const internal::GpuBufferSpec& spec);

  WebGpuTextureBufferPool(const wgpu::Instance& instance,
                          const wgpu::Device& device,
                          const internal::GpuBufferSpec& spec,
                          const MultiPoolOptions& options)
      : ReusablePool<WebGpuTextureBuffer>(
            [this] {
              return WebGpuTextureBuffer::Create(
                  instance_, device_, spec_.width, spec_.height, spec_.format);
            },
            options),
        instance_(instance),
        device_(device),
        spec_(spec) {}

 private:
  const wgpu::Instance instance_;
  const wgpu::Device device_;
  const internal::GpuBufferSpec& spec_;
};

class WebGpuTextureBufferMultiPool
    : public MultiPool<WebGpuTextureBufferPool, internal::GpuBufferSpec,
                       std::shared_ptr<WebGpuTextureBuffer>> {
 public:
  explicit WebGpuTextureBufferMultiPool(
      const wgpu::Instance& instance, const wgpu::Device& device,
      const MultiPoolOptions& options = kDefaultMultiPoolOptions)
      : MultiPool(
            [this](const internal::GpuBufferSpec& spec,
                   const MultiPoolOptions& options) {
              return WebGpuTextureBufferPool::Create(instance_, device_, spec,
                                                     options);
            },
            options),
        instance_(instance),
        device_(device) {}

  // TODO: This and RenderToWebGpuCanvas use kBGRA32 as defaults, while
  // the rest of our code defaults to kRGBA32 instead. Investigate whether this
  // is intentional or typos leading to accidental over-sharding of pools.
  absl::StatusOr<std::shared_ptr<WebGpuTextureBuffer>> GetBuffer(
      int width, int height,
      GpuBufferFormat format = GpuBufferFormat::kBGRA32) {
    return Get(internal::GpuBufferSpec(width, height, format));
  }

 private:
  const wgpu::Instance instance_;
  const wgpu::Device device_;
};

inline const WebGpuContextAttachment<WebGpuTextureBufferMultiPool>
    kWebGpuTexturePool{[](WebGpuContext& ctx) {
      return WebGpuContextAttachment<WebGpuTextureBufferMultiPool>::MakePtr(
          ctx.instance, ctx.device);
    }};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_WEBGPU_WEBGPU_TEXTURE_BUFFER_H_
