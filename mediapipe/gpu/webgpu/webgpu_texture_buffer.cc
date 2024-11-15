#include "mediapipe/gpu/webgpu/webgpu_texture_buffer.h"

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"

namespace mediapipe {

static wgpu::Texture CreateTexture(const wgpu::Device& device, uint32_t width,
                                   uint32_t height, GpuBufferFormat format,
                                   wgpu::TextureUsage extra_usage) {
  wgpu::TextureFormat wgpu_format;
  switch (format) {
    case GpuBufferFormat::kBGRA32:
      wgpu_format = wgpu::TextureFormat::RGBA8Unorm;
      break;
    case GpuBufferFormat::kRGBA32:
      wgpu_format = wgpu::TextureFormat::RGBA8Unorm;
      break;
    case GpuBufferFormat::kRGBAFloat128:
      wgpu_format = wgpu::TextureFormat::RGBA32Float;
      break;
    case GpuBufferFormat::kGrayFloat32:
      wgpu_format = wgpu::TextureFormat::R32Float;
      break;
    default:
      // We leave default the same, to ensure we don't break ongoing WebGPU
      // experiment. But we leave in one log statement so we can tell if this is
      // ever occurring.
      wgpu_format = wgpu::TextureFormat::RGBA8Unorm;
      ABSL_LOG_FIRST_N(WARNING, 1) << "WebGpuTextureBuffer created with "
                                   << "non-supported GpuBuffer format type: "
                                   << static_cast<uint32_t>(format) << ". "
                                   << "Defaulting to RGBA8Unorm.";
  }
  const wgpu::TextureDescriptor desc = {
      .nextInChain = nullptr,
      .label = nullptr,
      .usage = wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst |
               wgpu::TextureUsage::TextureBinding |
               wgpu::TextureUsage::StorageBinding |
               wgpu::TextureUsage::RenderAttachment | extra_usage,
      .dimension = wgpu::TextureDimension::e2D,
      .size =
          {
              .width = width,
              .height = height,
              .depthOrArrayLayers = 1,
          },
      .format = wgpu_format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  return device.CreateTexture(&desc);
}

std::unique_ptr<WebGpuTextureBuffer> WebGpuTextureBuffer::Create(
    const wgpu::Device& device, uint32_t width, uint32_t height,
    GpuBufferFormat format) {
  if (format != GpuBufferFormat::kRGBA32 &&
      format != GpuBufferFormat::kRGBAFloat128 &&
      format != GpuBufferFormat::kGrayFloat32)
    return nullptr;
  wgpu::Texture texture = CreateTexture(device, width, height, format, {});
  return std::make_unique<WebGpuTextureBuffer>(std::move(texture), width,
                                               height, format);
}

std::unique_ptr<WebGpuTextureBuffer> WebGpuTextureBuffer::Create(
    uint32_t width, uint32_t height, GpuBufferFormat format) {
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  if (!cc) return nullptr;
  const wgpu::Device device = cc->Service(kWebGpuService).GetObject().device();
  return Create(device, width, height, format);
}

WebGpuTextureView WebGpuTextureBuffer::GetReadView(
    internal::types<WebGpuTextureView>) const {
  return WebGpuTextureView(texture_, width_, height_);
}

WebGpuTextureView WebGpuTextureBuffer::GetWriteView(
    internal::types<WebGpuTextureView>) {
  return WebGpuTextureView(texture_, width_, height_);
}

absl::StatusOr<std::shared_ptr<WebGpuTextureBuffer>>
WebGpuTextureBufferPool::CreateBufferWithoutPool(
    const internal::GpuBufferSpec& spec) {
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  RET_CHECK(cc) << "Calculator context not found.";
  const wgpu::Device device = cc->Service(kWebGpuService).GetObject().device();
  std::unique_ptr<WebGpuTextureBuffer> buffer =
      WebGpuTextureBuffer::Create(device, spec.width, spec.height, spec.format);
  RET_CHECK(buffer) << absl::StrFormat(
      "Failed to Create WebGPU buffer: %d x %d, %d", spec.width, spec.height,
      static_cast<uint32_t>(spec.format));
  return buffer;
}

static std::shared_ptr<WebGpuTextureBuffer> GetWebGpuTextureBufferFromPool(
    int width, int height, GpuBufferFormat format) {
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  // TODO: gkarpiak - consider converting to ABSL_CHECK or better convert the
  // function to return absl::StatusOr.
  if (!cc) return nullptr;
  const wgpu::Device device = cc->Service(kWebGpuService).GetObject().device();
  auto& pool = GetWebGpuDeviceCachedAttachment(device, kWebGpuTexturePool);
  auto texture_buffer = pool.GetBuffer(width, height, format);
  ABSL_CHECK_OK(texture_buffer);
  return *texture_buffer;
}

static auto kWebGpuBufferPoolRegistration = [] {
  // Ensure that the WebGpuTextureBuffer's own factory is already registered,
  // so we can override it.
  WebGpuTextureBuffer::RegisterOnce();
  return internal::GpuBufferStorageRegistry::Get()
      .RegisterFactory<WebGpuTextureBuffer>(GetWebGpuTextureBufferFromPool);
}();

}  // namespace mediapipe
