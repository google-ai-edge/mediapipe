#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"

#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/objc/util.h"

namespace mediapipe {

#if TARGET_OS_OSX
typedef CVOpenGLTextureRef CVTextureType;
#else
typedef CVOpenGLESTextureRef CVTextureType;
#endif  // TARGET_OS_OSX

GlTextureView GpuBufferStorageCvPixelBuffer::GetReadView(
    mediapipe::internal::types<GlTextureView>,
    std::shared_ptr<GpuBuffer> gpu_buffer, int plane) const {
  CVReturn err;
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);
#if TARGET_OS_OSX
  CVTextureType cv_texture_temp;
  err = CVOpenGLTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), **this, NULL,
      &cv_texture_temp);
  CHECK(cv_texture_temp && !err)
      << "CVOpenGLTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLTextureGetTarget(*cv_texture),
      CVOpenGLTextureGetName(*cv_texture), width(), height(), *this, plane,
      [cv_texture](
          mediapipe::GlTextureView&) { /* only retains cv_texture */ });
#else
  const GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format(), plane, gl_context->GetGlVersion());
  CVTextureType cv_texture_temp;
  err = CVOpenGLESTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(), **this, NULL,
      GL_TEXTURE_2D, info.gl_internal_format, width() / info.downscale,
      height() / info.downscale, info.gl_format, info.gl_type, plane,
      &cv_texture_temp);
  CHECK(cv_texture_temp && !err)
      << "CVOpenGLESTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLESTextureGetTarget(*cv_texture),
      CVOpenGLESTextureGetName(*cv_texture), width(), height(),
      std::move(gpu_buffer), plane,
      [cv_texture](mediapipe::GlTextureView&) { /* only retains cv_texture */ },
      // TODO: make GetGlTextureView for write view non-const, remove cast
      // Note: we have to copy *this here because this storage is currently
      // stored in GpuBuffer by value, and so the this pointer becomes invalid
      // if the GpuBuffer is moved/copied. TODO: fix this.
      [me = *this](const mediapipe::GlTextureView& view) {
        const_cast<GpuBufferStorageCvPixelBuffer*>(&me)->ViewDoneWriting(view);
      });
#endif  // TARGET_OS_OSX
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetWriteView(
    mediapipe::internal::types<GlTextureView>,
    std::shared_ptr<GpuBuffer> gpu_buffer, int plane) {
  // For this storage there is currently no difference between read and write
  // views, so we delegate to the read method.
  return GetReadView(mediapipe::internal::types<GlTextureView>{},
                     std::move(gpu_buffer), plane);
}

void GpuBufferStorageCvPixelBuffer::ViewDoneWriting(const GlTextureView& view) {
#if TARGET_IPHONE_SIMULATOR
  CVPixelBufferRef pixel_buffer = **this;
  CHECK(pixel_buffer);
  CVReturn err = CVPixelBufferLockBaseAddress(pixel_buffer, 0);
  CHECK(err == kCVReturnSuccess)
      << "CVPixelBufferLockBaseAddress failed: " << err;
  OSType pixel_format = CVPixelBufferGetPixelFormatType(pixel_buffer);
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pixel_buffer);
  uint8_t* pixel_ptr =
      static_cast<uint8_t*>(CVPixelBufferGetBaseAddress(pixel_buffer));
  if (pixel_format == kCVPixelFormatType_32BGRA) {
    // TODO: restore previous framebuffer? Move this to helper so we
    // can use BindFramebuffer?
    glViewport(0, 0, view.width(), view.height());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                           view.name(), 0);

    size_t contiguous_bytes_per_row = view.width() * 4;
    if (bytes_per_row == contiguous_bytes_per_row) {
      glReadPixels(0, 0, view.width(), view.height(), GL_BGRA, GL_UNSIGNED_BYTE,
                   pixel_ptr);
    } else {
      std::vector<uint8_t> contiguous_buffer(contiguous_bytes_per_row *
                                             view.height());
      uint8_t* temp_ptr = contiguous_buffer.data();
      glReadPixels(0, 0, view.width(), view.height(), GL_BGRA, GL_UNSIGNED_BYTE,
                   temp_ptr);
      for (int i = 0; i < view.height(); ++i) {
        memcpy(pixel_ptr, temp_ptr, contiguous_bytes_per_row);
        temp_ptr += contiguous_bytes_per_row;
        pixel_ptr += bytes_per_row;
      }
    }
  } else {
    LOG(ERROR) << "unsupported pixel format: " << pixel_format;
  }
  err = CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
  CHECK(err == kCVReturnSuccess)
      << "CVPixelBufferUnlockBaseAddress failed: " << err;
#endif
}

std::unique_ptr<ImageFrame> GpuBufferStorageCvPixelBuffer::AsImageFrame()
    const {
  return CreateImageFrameForCVPixelBuffer(**this);
}

}  // namespace mediapipe
