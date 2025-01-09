#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"

#include <memory>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"
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

GlTextureView GpuBufferStorageCvPixelBuffer::GetTexture(
    int plane, GlTextureView::DoneWritingFn done_writing) const {
  CVReturn err;
  auto gl_context = GlContext::GetCurrent();
  ABSL_CHECK(gl_context);
#if TARGET_OS_OSX
  CVTextureType cv_texture_temp;
  err = CVOpenGLTextureCacheCreateTextureFromImage(
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
#else
  const GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format(), plane, gl_context->GetGlVersion());
  CVTextureType cv_texture_temp;
  err = CVOpenGLESTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), **this, NULL,
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

#if TARGET_IPHONE_SIMULATOR
static void ViewDoneWritingSimulatorWorkaround(CVPixelBufferRef pixel_buffer,
                                               const GlTextureView& view) {
  ABSL_CHECK(pixel_buffer);
  auto ctx = GlContext::GetCurrent().get();
  if (!ctx) ctx = view.gl_context();
  ctx->Run([pixel_buffer, &view, ctx] {
    CVReturn err = CVPixelBufferLockBaseAddress(pixel_buffer, 0);
    ABSL_CHECK(err == kCVReturnSuccess)
        << "CVPixelBufferLockBaseAddress failed: " << err;
    OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
    size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
    uint8_t* pixel_ptr =
        static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixel_buffer));
    if (pixel_format == kCVPixelFormatType_32BGRA) {
      glBindFramebuffer(GL_FRAMEBUFFER, kUtilityFramebuffer.Get(*ctx));
      glViewport(0, 0, view.width(), view.height());
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             view.target(), view.name(), 0);

      size_t contiguous_bytes_per_row = view.width() * 4;
      if (bytes_per_row == contiguous_bytes_per_row) {
        glReadPixels(0, 0, view.width(), view.height(), GL_BGRA,
                     GL_UNSIGNED_BYTE, pixel_ptr);
      } else {
        // TODO: use GL_PACK settings for row length. We can expect
        // GLES 3.0 on iOS now.
        std::vector<uint8_t> contiguous_buffer(contiguous_bytes_per_row *
                                               view.height());
        uint8_t* temp_ptr = contiguous_buffer.data();
        glReadPixels(0, 0, view.width(), view.height(), GL_BGRA,
                     GL_UNSIGNED_BYTE, temp_ptr);
        for (int i = 0; i < view.height(); ++i) {
          memcpy(pixel_ptr, temp_ptr, contiguous_bytes_per_row);
          temp_ptr += contiguous_bytes_per_row;
          pixel_ptr += bytes_per_row;
        }
      }
      // TODO: restore previous framebuffer?
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             view.target(), 0, 0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {
      ABSL_LOG(ERROR) << "unsupported pixel format: " << pixel_format;
    }
    err = CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
    ABSL_CHECK(err == kCVReturnSuccess)
        << "CVPixelBufferUnlockBaseAddress failed: " << err;
  });
}
#endif  // TARGET_IPHONE_SIMULATOR

GlTextureView GpuBufferStorageCvPixelBuffer::GetWriteView(
    internal::types<GlTextureView>, int plane) {
  return GetTexture(plane,
#if TARGET_IPHONE_SIMULATOR
                    [pixel_buffer = CFHolder<CVPixelBufferRef>(*this)](
                        const mediapipe::GlTextureView& view) {
                      ViewDoneWritingSimulatorWorkaround(*pixel_buffer, view);
                    }
#else
      nullptr
#endif  // TARGET_IPHONE_SIMULATOR
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
