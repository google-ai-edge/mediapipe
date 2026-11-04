#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"

#include <TargetConditionals.h>  // IWYU pragma: keep

#if defined(__APPLE__)
#include <CoreFoundation/CFBase.h>             // IWYU pragma: keep
#include <CoreVideo/CVPixelBuffer.h>           // IWYU pragma: keep
#include <CoreVideo/CVPixelBufferIOSurface.h>  // IWYU pragma: keep
#include <CoreVideo/CVReturn.h>                // IWYU pragma: keep
#if !TARGET_OS_OSX
#include <CoreVideo/CVOpenGLESTexture.h>       // IWYU pragma: keep
#include <CoreVideo/CVOpenGLESTextureCache.h>  // IWYU pragma: keep
#include <OpenGLES/gltypes.h>                  // IWYU pragma: keep
#endif                                         // !TARGET_OS_OSX
#endif                                         // defined(__APPLE__)

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"    // from @com_google_absl
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_format.h"  // IWYU pragma: keep (MEDIAPIPE_GPU_BUFFER_POOL_USE_CV_PIXEL_BUFFER)
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"
#include "mediapipe/objc/CFHolder.h"
#include "mediapipe/objc/util.h"

namespace mediapipe {

#if TARGET_OS_OSX
typedef CVOpenGLTextureRef CVTextureType;
#else
typedef CVOpenGLESTextureRef CVTextureType;
#endif  // TARGET_OS_OSX

GpuBufferStorageCvPixelBuffer::GpuBufferStorageCvPixelBuffer(
    int width, int height, GpuBufferFormat format) {
  OSType cv_format = CVPixelFormatForGpuBufferFormat(format);
  ABSL_CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err =
      CreateCVPixelBufferWithoutPool(width, height, cv_format, &buffer);
  ABSL_CHECK(!err) << "Error creating pixel buffer: " << err;
  adopt(buffer);
}

#if TARGET_OS_SIMULATOR
// Creates a temporary byte-backed CVPixelBuffer wrapping the CPU-mapped base
// address(es) of `pixel_buffer`. The caller must keep `pixel_buffer` locked via
// CVPixelBufferLockBaseAddress for the lifetime of the returned pixel buffer
// and any OpenGL ES textures created from it.
static CFHolder<CVPixelBufferRef> CreateByteBackedPixelBuffer(
    CVPixelBufferRef pixel_buffer) {
  const size_t width = CVPixelBufferGetWidth(pixel_buffer);
  const size_t height = CVPixelBufferGetHeight(pixel_buffer);
  const OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  CVPixelBufferRef temp_pixel_buffer_raw = nullptr;
  CVReturn err = kCVReturnSuccess;
  if (CVPixelBufferIsPlanar(pixel_buffer)) {
    const size_t plane_count = CVPixelBufferGetPlaneCount(pixel_buffer);
    std::vector<void*> plane_base_addresses(plane_count);
    std::vector<size_t> plane_widths(plane_count);
    std::vector<size_t> plane_heights(plane_count);
    std::vector<size_t> plane_bytes_per_row(plane_count);
    for (size_t i = 0; i < plane_count; ++i) {
      void* plane_addr = CVPixelBufferGetBaseAddressOfPlane(pixel_buffer, i);
      ABSL_CHECK(plane_addr != nullptr);
      plane_base_addresses[i] = plane_addr;
      plane_widths[i] = CVPixelBufferGetWidthOfPlane(pixel_buffer, i);
      plane_heights[i] = CVPixelBufferGetHeightOfPlane(pixel_buffer, i);
      plane_bytes_per_row[i] =
          CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, i);
    }
    err = CVPixelBufferCreateWithPlanarBytes(
        kCFAllocatorDefault, width, height, pixel_format, /*dataPtr=*/nullptr,
        /*dataSize=*/0, plane_count, plane_base_addresses.data(),
        plane_widths.data(), plane_heights.data(), plane_bytes_per_row.data(),
        /*releaseCallback=*/nullptr, /*releaseRefCon=*/nullptr,
        /*pixelBufferAttributes=*/nullptr, &temp_pixel_buffer_raw);
  } else {
    void* base_address = CVPixelBufferGetBaseAddress(pixel_buffer);
    ABSL_CHECK(base_address != nullptr);
    err = CVPixelBufferCreateWithBytes(
        kCFAllocatorDefault, width, height, pixel_format, base_address,
        CVPixelBufferGetBytesPerRow(pixel_buffer), /*releaseCallback=*/nullptr,
        /*releaseRefCon=*/nullptr, /*pixelBufferAttributes=*/nullptr,
        &temp_pixel_buffer_raw);
  }
  ABSL_CHECK(temp_pixel_buffer_raw && !err)
      << "CVPixelBufferCreateWithBytes failed: " << err;
  CFHolder<CVPixelBufferRef> temp_pixel_buffer;
  temp_pixel_buffer.adopt(temp_pixel_buffer_raw);
  return temp_pixel_buffer;
}

static void UnlockPixelBuffer(CVPixelBufferRef pixel_buffer,
                              CVPixelBufferLockFlags lock_flags) {
  CVReturn err = CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags);
  ABSL_CHECK(err == kCVReturnSuccess)
      << "CVPixelBufferUnlockBaseAddress failed: " << err;
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetTextureSimulator(
    const std::shared_ptr<GlContext>& gl_context, int plane,
    GlTextureView::DoneWritingFn done_writing) const {
  const GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format(), plane, gl_context->GetGlVersion());
  CVPixelBufferRef pixel_buffer_for_gl = **this;
  CFHolder<CVPixelBufferRef> temp_pixel_buffer;
  const CVPixelBufferLockFlags lock_flags =
      done_writing ? 0 : kCVPixelBufferLock_ReadOnly;
  bool locked = false;
  if (CVPixelBufferGetIOSurface(**this) != nullptr) {
    CVReturn lock_err = CVPixelBufferLockBaseAddress(**this, lock_flags);
    ABSL_CHECK(lock_err == kCVReturnSuccess)
        << "CVPixelBufferLockBaseAddress failed: " << lock_err;
    locked = true;
    temp_pixel_buffer = CreateByteBackedPixelBuffer(**this);
    pixel_buffer_for_gl = *temp_pixel_buffer;
  }

  CVTextureType cv_texture_temp = nullptr;
  CVReturn err = CVOpenGLESTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), pixel_buffer_for_gl,
      nullptr, GL_TEXTURE_2D, info.gl_internal_format, width() / info.downscale,
      height() / info.downscale, info.gl_format, info.gl_type, plane,
      &cv_texture_temp);
  if (err != kCVReturnSuccess || !cv_texture_temp) {
    if (locked) {
      UnlockPixelBuffer(**this, lock_flags);
    }
    ABSL_LOG(ERROR) << "CVOpenGLESTextureCacheCreateTextureFromImage failed: "
                    << err;
    return GlTextureView();
  }

  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  if (plane >= 0 && plane < kMaxSimulatorPlanes) {
    sim_temp_pixel_buffers_[plane] = temp_pixel_buffer;
    sim_cv_textures_[plane] = cv_texture;
  }
  const GLenum target = CVOpenGLESTextureGetTarget(*cv_texture);
  const GLuint name = CVOpenGLESTextureGetName(*cv_texture);
  return GlTextureView(
      gl_context.get(), target, name, width(), height(), plane,
      [cv_texture = std::move(cv_texture),
       temp_pixel_buffer = std::move(temp_pixel_buffer),
       parent_buffer = CFHolder<CVPixelBufferRef>(*this), locked,
       lock_flags](mediapipe::GlTextureView&) mutable {
        if (locked) {
          UnlockPixelBuffer(*parent_buffer, lock_flags);
        }
      },
      std::move(done_writing));
}
#endif  // TARGET_OS_SIMULATOR

GlTextureView GpuBufferStorageCvPixelBuffer::GetTexture(
    int plane, GlTextureView::DoneWritingFn done_writing) const {
  auto gl_context = GlContext::GetCurrent();
  ABSL_CHECK(gl_context);
#if TARGET_OS_OSX
  CVTextureType cv_texture_temp;
  CVReturn err = CVOpenGLTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), **this, NULL,
      &cv_texture_temp);
  ABSL_CHECK(cv_texture_temp && !err)
      << "CVOpenGLTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLTextureGetTarget(*cv_texture),
      CVOpenGLTextureGetName(*cv_texture), width(), height(), plane,
      [cv_texture](mediapipe::GlTextureView&) { /* only retains cv_texture */ },
      done_writing);
#elif TARGET_OS_SIMULATOR
  // On the iOS Simulator, CVMetalTextureCacheCreateTextureFromImage requires
  // CVPixelBuffers to be IOSurface-backed, whereas
  // CVOpenGLESTextureCacheCreateTextureFromImage rejects IOSurface-backed pixel
  // buffers with kCVReturnPixelBufferNotOpenGLCompatible (-6683). To bridge the
  // two, lock the IOSurface-backed pixel buffer and wrap its CPU-mapped base
  // address(es) in a temporary byte-backed CVPixelBuffer for OpenGL ES, keeping
  // the backing buffer locked until the GlTextureView is released.
  return GetTextureSimulator(gl_context, plane, std::move(done_writing));
#else
  const GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format(), plane, gl_context->GetGlVersion());
  CVTextureType cv_texture_temp = nullptr;
  CVReturn err = CVOpenGLESTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), **this, nullptr,
      GL_TEXTURE_2D, info.gl_internal_format, width() / info.downscale,
      height() / info.downscale, info.gl_format, info.gl_type, plane,
      &cv_texture_temp);
  ABSL_CHECK(cv_texture_temp && !err)
      << "CVOpenGLESTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLESTextureGetTarget(*cv_texture),
      CVOpenGLESTextureGetName(*cv_texture), width(), height(), plane,
      [cv_texture](mediapipe::GlTextureView&) { /* only retains cv_texture */ },
      done_writing);
#endif  // TARGET_OS_OSX
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetReadView(
    internal::types<GlTextureView>, int plane) const {
  return GetTexture(plane, nullptr);
}

#if TARGET_OS_SIMULATOR
static size_t BytesPerPixelForGlTextureInfo(const GlTextureInfo& info) {
  size_t channels = 0;
  switch (info.gl_format) {
    case GL_RGBA:
    case GL_BGRA:
      channels = 4;
      break;
    case GL_RGB:
      channels = 3;
      break;
    case GL_RG:
      channels = 2;
      break;
    case GL_RED:
      channels = 1;
      break;
    default:
      ABSL_LOG(ERROR) << "unsupported gl_format: " << info.gl_format;
      return 0;
  }
  size_t channel_size = 0;
  switch (info.gl_type) {
    case GL_UNSIGNED_BYTE:
      channel_size = 1;
      break;
    case GL_HALF_FLOAT:
    case GL_HALF_FLOAT_OES:
      channel_size = 2;
      break;
    case GL_FLOAT:
      channel_size = 4;
      break;
    default:
      ABSL_LOG(ERROR) << "unsupported gl_type: " << info.gl_type;
      return 0;
  }
  return channels * channel_size;
}

static void ReadTextureIntoBuffer(GlContext* ctx, const GlTextureView& view,
                                  const GlTextureInfo& info,
                                  size_t bytes_per_pixel, size_t bytes_per_row,
                                  uint8_t* pixel_ptr) {
  const int plane_width = view.width() / info.downscale;
  const int plane_height = view.height() / info.downscale;
  GLint prev_framebuffer = 0;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, kUtilityFramebuffer.Get(*ctx));
  glViewport(0, 0, plane_width, plane_height);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                         view.name(), 0);
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (status != GL_FRAMEBUFFER_COMPLETE) {
    ABSL_LOG(ERROR) << "Framebuffer incomplete in simulator workaround: "
                    << status;
  } else {
    GLint prev_pack_alignment = 4;
    glGetIntegerv(GL_PACK_ALIGNMENT, &prev_pack_alignment);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    const size_t contiguous_bytes_per_row = plane_width * bytes_per_pixel;
    if (bytes_per_row == contiguous_bytes_per_row) {
      glReadPixels(0, 0, plane_width, plane_height, info.gl_format,
                   info.gl_type, pixel_ptr);
    } else {
      // TODO: use GL_PACK settings for row length. We can expect
      // GLES 3.0 on iOS now.
      std::vector<uint8_t> contiguous_buffer(contiguous_bytes_per_row *
                                             plane_height);
      uint8_t* temp_ptr = contiguous_buffer.data();
      glReadPixels(0, 0, plane_width, plane_height, info.gl_format,
                   info.gl_type, temp_ptr);
      for (int i = 0; i < plane_height; ++i) {
        memcpy(pixel_ptr, temp_ptr, contiguous_bytes_per_row);
        temp_ptr += contiguous_bytes_per_row;
        pixel_ptr += bytes_per_row;
      }
    }
    glPixelStorei(GL_PACK_ALIGNMENT, prev_pack_alignment);
  }
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(), 0,
                         0);
  glBindFramebuffer(GL_FRAMEBUFFER, static_cast<GLuint>(prev_framebuffer));
}

// On the iOS Simulator, CVOpenGLESTextureCacheCreateTextureFromImage uploads
// the CVPixelBuffer's CPU memory into an OpenGL ES texture on creation, but
// rendering to that texture does not automatically write modified pixels back
// to the underlying CVPixelBuffer (or its IOSurface). Read the rendered texture
// plane back into the CVPixelBuffer via glReadPixels when writing finishes.
static void ViewDoneWritingSimulatorWorkaround(CVPixelBufferRef pixel_buffer,
                                               const GlTextureView& view) {
  ABSL_CHECK(pixel_buffer);
  auto ctx = GlContext::GetCurrent().get();
  if (!ctx) ctx = view.gl_context();
  ctx->Run([pixel_buffer, &view, ctx] {
    OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
    GpuBufferFormat gpu_format = GpuBufferFormatForCVPixelFormat(pixel_format);
    if (gpu_format == GpuBufferFormat::kUnknown) {
      ABSL_LOG(ERROR) << "unsupported pixel format: " << pixel_format;
      return;
    }
    const GlTextureInfo& info = GlTextureInfoForGpuBufferFormat(
        gpu_format, view.plane(), ctx->GetGlVersion());
    const size_t bytes_per_pixel = BytesPerPixelForGlTextureInfo(info);
    if (bytes_per_pixel == 0) {
      return;
    }

    CVReturn err = CVPixelBufferLockBaseAddress(pixel_buffer, 0);
    ABSL_CHECK(err == kCVReturnSuccess)
        << "CVPixelBufferLockBaseAddress failed: " << err;
    size_t bytes_per_row;
    uint8_t* pixel_ptr;
    if (CVPixelBufferIsPlanar(pixel_buffer)) {
      bytes_per_row =
          CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, view.plane());
      pixel_ptr = static_cast<uint8_t*>(
          CVPixelBufferGetBaseAddressOfPlane(pixel_buffer, view.plane()));
    } else {
      bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
      pixel_ptr =
          static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixel_buffer));
    }
    ABSL_CHECK(pixel_ptr != nullptr);

    ReadTextureIntoBuffer(ctx, view, info, bytes_per_pixel, bytes_per_row,
                          pixel_ptr);
    UnlockPixelBuffer(pixel_buffer, 0);
  });
}
#endif  // TARGET_OS_SIMULATOR

GlTextureView GpuBufferStorageCvPixelBuffer::GetWriteView(
    internal::types<GlTextureView>, int plane) {
  return GetTexture(plane,
#if TARGET_OS_SIMULATOR
                    // On the iOS Simulator, rendering to a CVOpenGLESTexture
                    // does not automatically update the backing CVPixelBuffer
                    // memory, so read back the texture when writing finishes.
                    [pixel_buffer = CFHolder<CVPixelBufferRef>(*this)](
                        const mediapipe::GlTextureView& view) {
                      ViewDoneWritingSimulatorWorkaround(*pixel_buffer, view);
                    }
#else
      nullptr
#endif  // TARGET_OS_SIMULATOR
  );
}

std::shared_ptr<const ImageFrame> GpuBufferStorageCvPixelBuffer::GetReadView(
    internal::types<ImageFrame>) const {
  return CreateImageFrameForCVPixelBuffer(**this);
}
std::shared_ptr<ImageFrame> GpuBufferStorageCvPixelBuffer::GetWriteView(
    internal::types<ImageFrame>) {
  return CreateImageFrameForCVPixelBuffer(**this);
}

#if MEDIAPIPE_GPU_BUFFER_POOL_USE_CV_PIXEL_BUFFER
// Only register this converter where the GPU buffer pool actually vends
// CVPixelBuffer-backed storage. On macOS the pool vends GlTextureBuffer, and
// registering this converter would hijack ImageFrame -> GpuBuffer conversions
// and abort on formats CVPixelBuffer cannot represent (e.g. 3-channel SRGB).
static std::shared_ptr<GpuBufferStorageCvPixelBuffer> ConvertFromImageFrame(
    std::shared_ptr<GpuBufferStorageImageFrame> frame) {
  auto status_or_buffer =
      CreateCVPixelBufferForImageFrame(frame->image_frame());
  ABSL_CHECK_OK(status_or_buffer);
  return std::make_shared<GpuBufferStorageCvPixelBuffer>(
      std::move(status_or_buffer).value());
}

static auto kConverterFromImageFrameRegistration =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<GpuBufferStorageImageFrame,
                           GpuBufferStorageCvPixelBuffer>(
            ConvertFromImageFrame);
#endif  // MEDIAPIPE_GPU_BUFFER_POOL_USE_CV_PIXEL_BUFFER

namespace internal {
std::shared_ptr<internal::GpuBufferStorage> AsGpuBufferStorage(
    CFHolder<CVPixelBufferRef> pixel_buffer) {
  return std::make_shared<GpuBufferStorageCvPixelBuffer>(
      std::move(pixel_buffer));
}

std::shared_ptr<internal::GpuBufferStorage> AsGpuBufferStorage(
    CVPixelBufferRef pixel_buffer) {
  return std::make_shared<GpuBufferStorageCvPixelBuffer>(pixel_buffer);
}
}  // namespace internal

}  // namespace mediapipe
