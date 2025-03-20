#include <webgpu/webgpu_cpp.h>

#include <memory>

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/gpu/webgpu/webgpu_service.h"
#include "mediapipe/gpu/webgpu/webgpu_utils.h"

namespace mediapipe {

namespace {

constexpr wgpu::TextureUsage kTextureUsage =
    wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc |
    wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding |
    wgpu::TextureUsage::StorageBinding;

wgpu::TextureFormat GetWebGpuTextureFormat(Tensor::Shape shape,
                                           Tensor::ElementType element_type) {
  const int depth = BhwcDepthFromShape(shape);
  ABSL_QCHECK_LE(depth, 4)
      << "WebGpuTexture2dView supports only tensors with depth <= 4.";
  ABSL_QCHECK_EQ(BhwcBatchFromShape(shape), 1)
      << "WebGpuTexture2dView supports only tensors with batch = 1.";

  if (element_type == Tensor::ElementType::kFloat16) {
    // Pad all F16 data to RGBA because only RGBA textures support storage
    // binding.
    return wgpu::TextureFormat::RGBA16Float;
  } else if (element_type == Tensor::ElementType::kFloat32) {
    switch (depth) {
      case 1:
        return wgpu::TextureFormat::R32Float;
      case 2:
        return wgpu::TextureFormat::RG32Float;
      case 3:
        // Padding to RGB -> RGBA.
        return wgpu::TextureFormat::RGBA32Float;
      case 4:
        return wgpu::TextureFormat::RGBA32Float;
      default:
        ABSL_QCHECK(false) << "Unsupported texture depth: " << depth;
    }
  } else {
    ABSL_CHECK(false)
        << "WebGpuTexture2dView supports only tensors with element type "
           "float16 or float32.";
  }
}

}  // namespace

Tensor::WebGpuTexture2dView Tensor::GetWebGpuTexture2dReadView(
    const WebGpuService& service) const {
  ABSL_QCHECK_NE(valid_, kValidNone)
      << "Tensor must be written prior to read from.";
  auto lock = std::make_unique<absl::MutexLock>(&view_mutex_);
  if (!(valid_ & kValidWebGpuTexture2d)) {
    ABSL_QCHECK(valid_ & kValidCpu)
        << "Cannot get a WebGPU read view into a tensor that is neither a "
           "valid CPU or WebGPU tensor.";
    const wgpu::Device& device = service.device();
    const wgpu::Queue& queue = device.GetQueue();

    const uint32_t bytes_per_pixel =
        element_size() * BhwcDepthFromShape(shape_);
    const wgpu::TextureFormat format =
        GetWebGpuTextureFormat(shape_, element_type_);

    const auto texture_or_error = CreateWebGpuTexture2dAndUploadData(
        device, BhwcWidthFromShape(shape_), BhwcHeightFromShape(shape_), format,
        kTextureUsage, queue, bytes_per_pixel, cpu_buffer_);
    ABSL_QCHECK(texture_or_error.ok())
        << "Failed to create WebGPU texture: " << texture_or_error.status();
    webgpu_device_ = device;
    webgpu_texture2d_ = texture_or_error.value();
    valid_ |= kValidWebGpuTexture2d;
  }
  return {webgpu_texture2d_, std::move(lock)};
}

Tensor::WebGpuTexture2dView Tensor::GetWebGpuTexture2dWriteView(
    const WebGpuService& service) const {
  const wgpu::Device& device = service.device();
  ABSL_QCHECK(device)
      << "WebGpuTexture2dView: a valid wgpu device must be provided.";
  auto lock = std::make_unique<absl::MutexLock>(&view_mutex_);
  // TODO: MLDrift expects 4-channel textures for writing output, this
  // may be possible to change in the future.
  wgpu::TextureFormat format;
  if (element_type_ == Tensor::ElementType::kFloat16) {
    format = wgpu::TextureFormat::RGBA16Float;
  } else if (element_type_ == Tensor::ElementType::kFloat32) {
    format = wgpu::TextureFormat::RGBA32Float;
  } else {
    ABSL_QCHECK(false)
        << "WebGpuTexture2dView supports only tensors with element type "
           "float16 or float32.";
  }

  if (!webgpu_texture2d_) {
    webgpu_device_ = device;
    webgpu_texture2d_ = CreateTextureWebGpuTexture2d(
        device, BhwcWidthFromShape(shape_), BhwcHeightFromShape(shape_), format,
        kTextureUsage);
  }
  valid_ = kValidWebGpuTexture2d;
  return {webgpu_texture2d_, std::move(lock)};
}

}  // namespace mediapipe
