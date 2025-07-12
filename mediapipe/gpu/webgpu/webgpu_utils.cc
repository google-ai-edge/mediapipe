#include "mediapipe/gpu/webgpu/webgpu_utils.h"

#include <webgpu/webgpu_cpp.h>
#ifdef __EMSCRIPTEN__
#include <emscripten/em_js.h>
#include <emscripten/emscripten.h>
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/web/jspi_check.h"

namespace mediapipe {

namespace {

static absl::NoDestructor<wgpu::Instance> kWebGpuInstance([] {
  wgpu::InstanceDescriptor instance_desc = {};
  static const auto kTimedWaitAny = wgpu::InstanceFeatureName::TimedWaitAny;
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = &kTimedWaitAny;
  return wgpu::CreateInstance(&instance_desc);
}());

#ifdef __EMSCRIPTEN__

EM_ASYNC_JS(void, mediapipe_map_buffer_jspi,
            (WGPUBuffer buffer_handle, uint8_t* data), {
              const buffer = WebGPU.getJsObject(buffer_handle);
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

template <typename T>
WebGpuAsyncFuture<T>::WebGpuAsyncFuture(WebGpuAsyncFuture<T>&& other)
    : future_(other.future_), result_(std::move(other.result_)) {
  other.future_ = std::nullopt;
}

template <typename T>
WebGpuAsyncFuture<T>::~WebGpuAsyncFuture() {
  Reset();
}

template <typename T>
WebGpuAsyncFuture<T>& WebGpuAsyncFuture<T>::operator=(
    WebGpuAsyncFuture<T>&& other) {
  Reset();  // Free the current future if any.
  future_ = other.future_;
  other.future_ = std::nullopt;
  result_ = std::move(other.result_);
  return *this;
}

template <typename T>
absl::StatusOr<T*> WebGpuAsyncFuture<T>::Get(absl::Duration timeout) {
  if (result_ == nullptr) {
    return absl::FailedPreconditionError("Uninitialized WebGpuAsyncFuture.");
  }

  if (!result_->has_value()) {
    if (!future_.has_value()) {
      return absl::FailedPreconditionError("No value and no pending future.");
    }

    wgpu::WaitStatus wait_status = kWebGpuInstance->WaitAny(
        future_.value(), timeout == absl::InfiniteDuration()
                             ? UINT64_MAX
                             : absl::ToInt64Nanoseconds(timeout));
    if (wait_status == wgpu::WaitStatus::TimedOut) {
      return absl::DeadlineExceededError(
          "Timed out waiting for WebGPU future.");
    } else if (wait_status != wgpu::WaitStatus::Success) {
      return absl::InternalError("WebGPU future wait failed.");
    }
    future_ = std::nullopt;

    if (!result_->has_value()) {
      return absl::InternalError("Result not set.");
    }
  }

  auto& value = result_->value();
  if (value.ok()) {
    return absl::StatusOr<T*>(&value.value());
  } else {
    return value.status();
  }
}

template <typename T>
void WebGpuAsyncFuture<T>::Reset() {
  if (future_.has_value()) {
    // Collect the result of the future to avoid a memory leak.
    Get().IgnoreError();
  }
  future_ = std::nullopt;
  result_ = nullptr;
}

template class WebGpuAsyncFuture<wgpu::ComputePipeline>;
template class WebGpuAsyncFuture<wgpu::RenderPipeline>;

wgpu::ShaderModule CreateWgslShader(wgpu::Device device, const char* const code,
                                    const char* label) {
  wgpu::ShaderSourceWGSL wgsl;
  wgsl.code = code;
  wgpu::ShaderModuleDescriptor desc;
  desc.nextInChain = &wgsl;
  desc.label = label;
  return device.CreateShaderModule(&desc);
}

WebGpuAsyncFuture<wgpu::ComputePipeline> WebGpuCreateComputePipelineAsync(
    const wgpu::Device& device,
    wgpu::ComputePipelineDescriptor const* descriptor) {
  auto holder =
      std::make_unique<std::optional<absl::StatusOr<wgpu::ComputePipeline>>>();
  auto* holder_ptr = holder.get();

#ifdef __EMSCRIPTEN__
  if (!IsJspiAvailable()) {
    *holder_ptr = device.CreateComputePipeline(descriptor);
    return WebGpuAsyncFuture<wgpu::ComputePipeline>(std::nullopt,
                                                    std::move(holder));
  }
#endif  // __EMSCRIPTEN__

  auto future = device.CreateComputePipelineAsync(
      descriptor, wgpu::CallbackMode::WaitAnyOnly,
      [holder_ptr](wgpu::CreatePipelineAsyncStatus status,
                   wgpu::ComputePipeline pipeline, wgpu::StringView message) {
        if (status != wgpu::CreatePipelineAsyncStatus::Success) {
          *holder_ptr = absl::InternalError(std::string(message));
          return;
        }
        *holder_ptr = pipeline;
      });
  return WebGpuAsyncFuture<wgpu::ComputePipeline>(future, std::move(holder));
}

WebGpuAsyncFuture<wgpu::RenderPipeline> WebGpuCreateRenderPipelineAsync(
    const wgpu::Device& device,
    wgpu::RenderPipelineDescriptor const* descriptor) {
  auto holder =
      std::make_unique<std::optional<absl::StatusOr<wgpu::RenderPipeline>>>();
  auto* holder_ptr = holder.get();

#ifdef __EMSCRIPTEN__
  if (!IsJspiAvailable()) {
    *holder_ptr = device.CreateRenderPipeline(descriptor);
    return WebGpuAsyncFuture<wgpu::RenderPipeline>(std::nullopt,
                                                   std::move(holder));
  }
#endif  // __EMSCRIPTEN__

  auto future = device.CreateRenderPipelineAsync(
      descriptor, wgpu::CallbackMode::WaitAnyOnly,
      [holder_ptr](wgpu::CreatePipelineAsyncStatus status,
                   wgpu::RenderPipeline pipeline, wgpu::StringView message) {
        if (status != wgpu::CreatePipelineAsyncStatus::Success) {
          *holder_ptr = absl::InternalError(std::string(message));
          return;
        }
        *holder_ptr = pipeline;
      });
  return WebGpuAsyncFuture<wgpu::RenderPipeline>(future, std::move(holder));
}

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

  wgpu::TexelCopyTextureInfo destination = {.texture = texture};
  wgpu::TexelCopyBufferLayout texel_copy_buffer_layout = {
      .bytesPerRow = width * texture_bytes_per_pixel,
      .rowsPerImage = height,
  };
  wgpu::Extent3D write_size = {
      .width = width,
      .height = height,
  };

  queue.WriteTexture(&destination, buffer, buffer_size,
                     &texel_copy_buffer_layout, &write_size);
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
  wgpu::TexelCopyTextureInfo copy_src{.texture = texture};
  wgpu::TexelCopyBufferInfo copy_dst{.layout = {.bytesPerRow = bytes_per_row},
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
