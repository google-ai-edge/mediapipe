#ifndef MEDIAPIPE_GPU_WEBGPU_WEBGPU_UTILS_H_
#define MEDIAPIPE_GPU_WEBGPU_WEBGPU_UTILS_H_

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/time/time.h"

namespace mediapipe {

template <typename T>
class WebGpuAsyncFuture {
 public:
  WebGpuAsyncFuture<T>() = default;
  WebGpuAsyncFuture<T>(WebGpuAsyncFuture<T>&& other);
  ~WebGpuAsyncFuture();
  inline explicit WebGpuAsyncFuture(
      std::optional<wgpu::Future> future,
      std::unique_ptr<std::optional<absl::StatusOr<T>>> result)
      : future_(future), result_(std::move(result)) {}

  WebGpuAsyncFuture<T>& operator=(WebGpuAsyncFuture<T>&& other);

  absl::StatusOr<T*> Get(absl::Duration timeout = absl::InfiniteDuration());
  void Reset();

 private:
  std::optional<wgpu::Future> future_;
  std::unique_ptr<std::optional<absl::StatusOr<T>>> result_;
};

wgpu::ShaderModule CreateWgslShader(wgpu::Device device, const char* code,
                                    const char* label);
WebGpuAsyncFuture<wgpu::ComputePipeline> WebGpuCreateComputePipelineAsync(
    const wgpu::Device& device,
    wgpu::ComputePipelineDescriptor const* descriptor);
WebGpuAsyncFuture<wgpu::RenderPipeline> WebGpuCreateRenderPipelineAsync(
    const wgpu::Device& device,
    wgpu::RenderPipelineDescriptor const* descriptor);

absl::StatusOr<uint32_t> WebGpuTextureFormatBytesPerPixel(
    wgpu::TextureFormat format);
absl::StatusOr<uint32_t> WebGpuTextureFormatDepth(wgpu::TextureFormat format);

wgpu::Texture CreateTextureWebGpuTexture2d(const wgpu::Device& device,
                                           uint32_t width, uint32_t height,
                                           wgpu::TextureFormat format,
                                           wgpu::TextureUsage usage);

absl::Status WebGpuTexture2dUploadData(const wgpu::Device& device,
                                       uint32_t width, uint32_t height,
                                       wgpu::TextureFormat format,
                                       const wgpu::Queue& queue,
                                       uint32_t bytes_per_pixel, void* data,
                                       const wgpu::Texture& texture);

absl::StatusOr<wgpu::Texture> CreateWebGpuTexture2dAndUploadData(
    const wgpu::Device& device, uint32_t width, uint32_t height,
    wgpu::TextureFormat format, wgpu::TextureUsage usage,
    const wgpu::Queue& queue, uint32_t bytes_per_pixel, void* data);

#ifdef __EMSCRIPTEN__
absl::Status GetTexture2dData(const wgpu::Device& device,
                              const wgpu::Queue& queue,
                              const wgpu::Texture& texture, uint32_t width,
                              uint32_t height, uint32_t bytes_per_row,
                              uint8_t* dst);
#endif  // __EMSCRIPTEN__

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_WEBGPU_WEBGPU_UTILS_H_
