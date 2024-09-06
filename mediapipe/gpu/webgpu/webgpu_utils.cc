#include "mediapipe/gpu/webgpu/webgpu_utils.h"

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>
#ifdef __EMSCRIPTEN__
#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>
#endif

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/web/jspi_check.h"

namespace mediapipe {

namespace {

#ifdef __EMSCRIPTEN__

EM_ASYNC_JS(void, mediapipe_map_buffer_jspi,
            (WGPUBuffer buffer_handle, uint8_t* data), {
              const buffer = WebGPU.mgrBuffer.get(buffer_handle);
              await buffer.mapAsync(GPUMapMode.READ);
              const mapped = buffer.getMappedRange();
              HEAPU8.set(new Uint8Array(mapped), data);
              buffer.unmap();
            });

#endif  // __EMSCRIPTEN__

// WebGPU has no RGB texture format, so we pad the depth to 4 elements.
void PadElementDepth(uint8_t* src_buffer, uint8_t* dst_buffer, int num_elements,
                     int src_depth_bytes, int dst_depth_bytes) {
  for (int e = 0; e < num_elements; e++) {
    for (int i = 0; i < src_depth_bytes; i++) {
      dst_buffer[i] = src_buffer[i];
    }
    src_buffer += src_depth_bytes;
    dst_buffer += dst_depth_bytes;
  }
}

}  // namespace

absl::StatusOr<uint32_t> WebGpuTextureFormatBytesPerPixel(
    wgpu::TextureFormat format) {
  switch (format) {
    case wgpu::TextureFormat::RGBA8Unorm:
      return 4;
    case wgpu::TextureFormat::R16Float:
      return 2;
    case wgpu::TextureFormat::RG16Float:
      return 4;
    case wgpu::TextureFormat::RGBA16Float:
      return 8;
    case wgpu::TextureFormat::R32Float:
      return 4;
    case wgpu::TextureFormat::RG32Float:
      return 8;
    case wgpu::TextureFormat::RGBA32Float:
      return 16;
    default:
      return absl::InvalidArgumentError("Unsupported texture format.");
  }
}

absl::StatusOr<uint32_t> WebGpuTextureFormatDepth(wgpu::TextureFormat format) {
  switch (format) {
    case wgpu::TextureFormat::R16Float:
    case wgpu::TextureFormat::R32Float:
      return 1;
    case wgpu::TextureFormat::RG16Float:
    case wgpu::TextureFormat::RG32Float:
      return 2;
    case wgpu::TextureFormat::RGBA16Float:
    case wgpu::TextureFormat::RGBA32Float:
      return 4;
    default:
      return absl::InvalidArgumentError("Unsupported texture format.");
  }
}

wgpu::Texture CreateTextureWebGpuTexture2d(const wgpu::Device& device,
                                           uint32_t width, uint32_t height,
                                           wgpu::TextureFormat format,
                                           wgpu::TextureUsage usage) {
  wgpu::TextureDescriptor texture_desc;
  texture_desc.dimension = wgpu::TextureDimension::e2D;
  texture_desc.size.width = width;
  texture_desc.size.height = height;
  texture_desc.size.depthOrArrayLayers = 1;
  texture_desc.sampleCount = 1;
  texture_desc.format = format;
  texture_desc.mipLevelCount = 1;
  texture_desc.usage = usage;

  return device.CreateTexture(&texture_desc);
}

absl::Status WebGpuTexture2dUploadData(const wgpu::Device& device,
                                       uint32_t width, uint32_t height,
                                       wgpu::TextureFormat format,
                                       const wgpu::Queue& queue,
                                       uint32_t bytes_per_pixel, void* data,
                                       const wgpu::Texture& texture) {
  MP_ASSIGN_OR_RETURN(const uint32_t texture_bytes_per_pixel,
                      WebGpuTextureFormatBytesPerPixel(format));
  const uint32_t buffer_size = width * height * texture_bytes_per_pixel;
  std::unique_ptr<uint8_t[]> temp_buffer;
  void* buffer;

  if (bytes_per_pixel != texture_bytes_per_pixel) {
    temp_buffer = std::make_unique<uint8_t[]>(buffer_size);
    PadElementDepth(reinterpret_cast<uint8_t*>(data), temp_buffer.get(),
                    width * height, bytes_per_pixel, texture_bytes_per_pixel);
    buffer = temp_buffer.get();
  } else {
    buffer = data;
  }

  wgpu::ImageCopyTexture destination = {.texture = texture};
  wgpu::TextureDataLayout texture_data_layout = {
      .bytesPerRow = width * texture_bytes_per_pixel,
      .rowsPerImage = height,
  };
  wgpu::Extent3D write_size = {
      .width = width,
      .height = height,
  };

  queue.WriteTexture(&destination, buffer, buffer_size, &texture_data_layout,
                     &write_size);
  return absl::OkStatus();
}

absl::StatusOr<wgpu::Texture> CreateWebGpuTexture2dAndUploadData(
    const wgpu::Device& device, uint32_t width, uint32_t height,
    wgpu::TextureFormat format, wgpu::TextureUsage usage,
    const wgpu::Queue& queue, uint32_t bytes_per_pixel, void* data) {
  // wgpu::TextureUsage::CopyDst is required for the WriteTexture call.
  wgpu::Texture texture = CreateTextureWebGpuTexture2d(
      device, width, height, format, usage | wgpu::TextureUsage::CopyDst);
  MP_RETURN_IF_ERROR(WebGpuTexture2dUploadData(
      device, width, height, format, queue, bytes_per_pixel, data, texture));
  return texture;
}

#ifdef __EMSCRIPTEN__

// TODO: Implement when not using emscripten.
absl::Status GetTexture2dData(const wgpu::Device& device,
                              const wgpu::Queue& queue,
                              const wgpu::Texture& texture, uint32_t width,
                              uint32_t height, uint32_t bytes_per_row,
                              uint8_t* dst) {
  if (!IsJspiAvailable()) {
    return absl::UnimplementedError("GetTexture2dData requires JSPI.");
  }
  const uint32_t buffer_size = height * bytes_per_row;

  wgpu::BufferDescriptor buffer_descriptor = {
      .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
      .size = (uint64_t)buffer_size};
  wgpu::Buffer webgpu_buffer = device.CreateBuffer(&buffer_descriptor);

  auto command_encoder = device.CreateCommandEncoder({});
  wgpu::ImageCopyTexture copy_src{.texture = texture};
  wgpu::ImageCopyBuffer copy_dst{.layout = {.bytesPerRow = bytes_per_row},
                                 .buffer = webgpu_buffer};
  wgpu::Extent3D copy_size = {
      .width = width,
      .height = height,
  };

  command_encoder.CopyTextureToBuffer(&copy_src, &copy_dst, &copy_size);
  wgpu::CommandBuffer copy_command_buffer = command_encoder.Finish();
  queue.Submit(1, &copy_command_buffer);

  mediapipe_map_buffer_jspi(webgpu_buffer.Get(), dst);
  webgpu_buffer.Destroy();

  return absl::OkStatus();
}

#endif  // __EMSCRIPTEN__

}  // namespace mediapipe
