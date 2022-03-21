#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"

#include <memory>

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
  CHECK_NE(cv_format, -1) << "unsupported pixel format";
  CVPixelBufferRef buffer;
  CVReturn err =
      CreateCVPixelBufferWithoutPool(width, height, cv_format, &buffer);
  CHECK(!err) << "Error creating pixel buffer: " << err;
  adopt(buffer);
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetTexture(
    std::shared_ptr<GpuBuffer> gpu_buffer, int plane,
    GlTextureView::DoneWritingFn done_writing) const {
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
  CHECK(cv_texture_temp && !err)
      << "CVOpenGLESTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLESTextureGetTarget(*cv_texture),
      CVOpenGLESTextureGetName(*cv_texture), width(), height(),
      std::move(gpu_buffer), plane,
      [cv_texture](mediapipe::GlTextureView&) { /* only retains cv_texture */ },
      done_writing);
#endif  // TARGET_OS_OSX
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetReadView(
    internal::types<GlTextureView>, std::shared_ptr<GpuBuffer> gpu_buffer,
    int plane) const {
  return GetTexture(std::move(gpu_buffer), plane, nullptr);
}

GlTextureView GpuBufferStorageCvPixelBuffer::GetWriteView(
    internal::types<GlTextureView>, std::shared_ptr<GpuBuffer> gpu_buffer,
    int plane) {
  return GetTexture(
      std::move(gpu_buffer), plane,
      [this](const mediapipe::GlTextureView& view) { ViewDoneWriting(view); });
}

std::shared_ptr<const ImageFrame> GpuBufferStorageCvPixelBuffer::GetReadView(
    internal::types<ImageFrame>, std::shared_ptr<GpuBuffer> gpu_buffer) const {
  return CreateImageFrameForCVPixelBuffer(**this);
}
std::shared_ptr<ImageFrame> GpuBufferStorageCvPixelBuffer::GetWriteView(
    internal::types<ImageFrame>, std::shared_ptr<GpuBuffer> gpu_buffer) {
  return CreateImageFrameForCVPixelBuffer(**this);
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

static std::shared_ptr<GpuBufferStorageCvPixelBuffer> ConvertFromImageFrame(
    std::shared_ptr<GpuBufferStorageImageFrame> frame) {
  auto status_or_buffer =
      CreateCVPixelBufferForImageFrame(frame->image_frame());
  CHECK(status_or_buffer.ok());
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
