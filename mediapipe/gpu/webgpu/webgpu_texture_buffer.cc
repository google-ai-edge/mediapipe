#include "mediapipe/gpu/webgpu/webgpu_texture_buffer.h"

#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"      // from @com_google_absl
#include "absl/log/absl_log.h"        // from @com_google_absl
#include "absl/status/statusor.h"     // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/legacy_calculator_support.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"
#include "mediapipe/gpu/webgpu/webgpu_headers.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_texture_view.h"
#include "mediapipe/gpu/webgpu/webgpu_utils.h"

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
    const wgpu::Instance& instance, const wgpu::Device& device, uint32_t width,
    uint32_t height, GpuBufferFormat format) {
  if (format != GpuBufferFormat::kRGBA32 &&
      format != GpuBufferFormat::kRGBAFloat128 &&
      format != GpuBufferFormat::kGrayFloat32)
    return nullptr;
  wgpu::Texture texture = CreateTexture(device, width, height, format, {});
  return std::make_unique<WebGpuTextureBuffer>(
      instance, device, std::move(texture), width, height, format);
}

std::unique_ptr<WebGpuTextureBuffer> WebGpuTextureBuffer::Create(
    uint32_t width, uint32_t height, GpuBufferFormat format) {
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  if (!cc) return nullptr;
  const auto& service = cc->Service(kWebGpuService).GetObject();
  return Create(service.instance(), service.device(), width, height, format);
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
  const auto& service = cc->Service(kWebGpuService).GetObject();
  std::unique_ptr<WebGpuTextureBuffer> buffer =
      WebGpuTextureBuffer::Create(service.instance(), service.device(),
                                  spec.width, spec.height, spec.format);
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
  auto& pool =
      cc->Service(kWebGpuService).GetObject().GetAttachment(kWebGpuTexturePool);
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

static std::shared_ptr<WebGpuTextureBuffer> ConvertFromImageFrame(
    std::shared_ptr<GpuBufferStorageImageFrame> frame) {
  const auto cc = LegacyCalculatorSupport::Scoped<CalculatorContext>::current();
  ABSL_CHECK(cc);

  const uint32_t width = frame->width();
  const uint32_t height = frame->height();
  // texelCopyBufferInfo.bytesPerRow must be a multiple of 256.
  const uint32_t bytes_per_row = (width * 4 + 255) / 256 * 256;

  const auto& image_frame = frame->image_frame();
  const size_t buffer_size = bytes_per_row * height;
  ABSL_CHECK(image_frame->Format() == ImageFormat::SRGBA);

  WebGpuService& wgpu_service = cc->Service(kWebGpuService).GetObject();
  const wgpu::Device device = wgpu_service.device();
  const wgpu::BufferDescriptor buf_desc = {
      .usage = wgpu::BufferUsage::MapWrite | wgpu::BufferUsage::CopySrc,
      .size = buffer_size,
      .mappedAtCreation = true};
  auto buffer = device.CreateBuffer(&buf_desc);

  {
    const size_t data_row_bytes = width * 4;
    const char* img_data =
        reinterpret_cast<const char*>(image_frame->PixelData());
    auto* buf_data = static_cast<char*>(buffer.GetMappedRange(0, buffer_size));
    for (int i = 0; i < height; ++i) {
      memcpy(buf_data, img_data, data_row_bytes);
      img_data += image_frame->WidthStep();
      buf_data += bytes_per_row;
    }
  }
  buffer.Unmap();

  auto& pool = wgpu_service.GetAttachment(kWebGpuTexturePool);
  auto texture_buffer = pool.GetBuffer(width, height, GpuBufferFormat::kRGBA32);
  ABSL_CHECK_OK(texture_buffer.status());
  WebGpuTextureView view =
      (*texture_buffer)->GetWriteView(internal::types<WebGpuTextureView>());
  wgpu::Texture texture = view.texture();

  auto encoder = device.CreateCommandEncoder();
  const wgpu::TexelCopyBufferInfo src = {
      .layout = {.offset = 0,
                 .bytesPerRow = bytes_per_row,
                 .rowsPerImage = height},
      .buffer = buffer};
  const wgpu::TexelCopyTextureInfo dst = {
      .texture = texture,
      .mipLevel = 0,
      .origin = {0, 0, 0},
      .aspect = wgpu::TextureAspect::All,
  };
  const wgpu::Extent3D copy_size = {width, height, 1};
  encoder.CopyBufferToTexture(&src, &dst, &copy_size);
  wgpu::CommandBuffer command_buffers[] = {
      encoder.Finish(),
  };
  device.GetQueue().Submit(std::size(command_buffers), command_buffers);

  return *texture_buffer;
}

static auto kConverterFromImageFrameRegistration =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<GpuBufferStorageImageFrame, WebGpuTextureBuffer>(
            ConvertFromImageFrame);

static std::shared_ptr<GpuBufferStorageImageFrame> ConvertToImageFrame(
    std::shared_ptr<WebGpuTextureBuffer> frame) {
  const uint32_t width = frame->width();
  const uint32_t height = frame->height();
  const GpuBufferFormat format = frame->format();
  const ImageFormat::Format image_format =
      ImageFormatForGpuBufferFormat(format);

  // WebGpu readback requires 256-byte row alignment.
  auto out_image_frame =
      std::make_shared<ImageFrame>(image_format, width, height, 256);

  auto view = frame->GetReadView(internal::types<WebGpuTextureView>());

  ABSL_CHECK_OK(GetTexture2dData(frame->instance(), frame->device(),
                                 frame->device().GetQueue(), view.texture(),
                                 width, height, out_image_frame->WidthStep(),
                                 out_image_frame->MutablePixelData()));

  return std::make_shared<GpuBufferStorageImageFrame>(
      std::move(out_image_frame));
}

static auto kConverterToImageFrameRegistration =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<WebGpuTextureBuffer, GpuBufferStorageImageFrame>(
            ConvertToImageFrame);

}  // namespace mediapipe
