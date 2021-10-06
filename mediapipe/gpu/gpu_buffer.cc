#include "mediapipe/gpu/gpu_buffer.h"

#include "mediapipe/gpu/gl_context.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/objc/util.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

void GlTextureView::Release() {
  if (detach_) detach_(*this);
  detach_ = nullptr;
  gl_context_ = nullptr;
  gpu_buffer_ = nullptr;
  plane_ = 0;
  name_ = 0;
  width_ = 0;
  height_ = 0;
}

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#if TARGET_OS_OSX
typedef CVOpenGLTextureRef CVTextureType;
#else
typedef CVOpenGLESTextureRef CVTextureType;
#endif  // TARGET_OS_OSX

GlTextureView GpuBuffer::GetGlTextureView(int plane, bool for_reading) const {
  CVReturn err;
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);
#if TARGET_OS_OSX
  CVTextureType cv_texture_temp;
  err = CVOpenGLTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault, gl_context->cv_texture_cache(),
      GetCVPixelBufferRef(), NULL, &cv_texture_temp);
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
      kCFAllocatorDefault, gl_context->cv_texture_cache(),
      GetCVPixelBufferRef(), NULL, GL_TEXTURE_2D, info.gl_internal_format,
      width() / info.downscale, height() / info.downscale, info.gl_format,
      info.gl_type, plane, &cv_texture_temp);
  CHECK(cv_texture_temp && !err)
      << "CVOpenGLESTextureCacheCreateTextureFromImage failed: " << err;
  CFHolder<CVTextureType> cv_texture;
  cv_texture.adopt(cv_texture_temp);
  return GlTextureView(
      gl_context.get(), CVOpenGLESTextureGetTarget(*cv_texture),
      CVOpenGLESTextureGetName(*cv_texture), width(), height(), *this, plane,
      [cv_texture](
          mediapipe::GlTextureView&) { /* only retains cv_texture */ });
#endif  // TARGET_OS_OSX
}

GpuBuffer GpuBuffer::CopyingImageFrame(const ImageFrame& image_frame) {
  auto maybe_buffer = CreateCVPixelBufferCopyingImageFrame(image_frame);
  // Converts absl::StatusOr to absl::Status since CHECK_OK() currently only
  // deals with absl::Status in MediaPipe OSS.
  CHECK_OK(maybe_buffer.status());
  return GpuBuffer(std::move(maybe_buffer).value());
}

std::unique_ptr<ImageFrame> GpuBuffer::AsImageFrame() const {
  CHECK(GetCVPixelBufferRef());
  return CreateImageFrameForCVPixelBuffer(GetCVPixelBufferRef());
}

void GlTextureView::DoneWriting() const {
  CHECK(gpu_buffer_);
#if TARGET_IPHONE_SIMULATOR
  CVPixelBufferRef pixel_buffer = gpu_buffer_.GetCVPixelBufferRef();
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
    glViewport(0, 0, width(), height());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target(),
                           name(), 0);

    size_t contiguous_bytes_per_row = width() * 4;
    if (bytes_per_row == contiguous_bytes_per_row) {
      glReadPixels(0, 0, width(), height(), GL_BGRA, GL_UNSIGNED_BYTE,
                   pixel_ptr);
    } else {
      std::vector<uint8_t> contiguous_buffer(contiguous_bytes_per_row *
                                             height());
      uint8_t* temp_ptr = contiguous_buffer.data();
      glReadPixels(0, 0, width(), height(), GL_BGRA, GL_UNSIGNED_BYTE,
                   temp_ptr);
      for (int i = 0; i < height(); ++i) {
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
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

#if !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
GlTextureView GpuBuffer::GetGlTextureView(int plane, bool for_reading) const {
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);
  const GlTextureBufferSharedPtr& texture_buffer =
      GetGlTextureBufferSharedPtr();
  // Insert wait call to sync with the producer.
  texture_buffer->WaitOnGpu();
  CHECK_EQ(plane, 0);
  GlTextureView::DetachFn detach;
  if (for_reading) {
    detach = [](mediapipe::GlTextureView& texture) {
      // Inform the GlTextureBuffer that we have finished accessing its
      // contents, and create a consumer sync point.
      texture.gpu_buffer().GetGlTextureBufferSharedPtr()->DidRead(
          texture.gl_context()->CreateSyncToken());
    };
  }
  return GlTextureView(gl_context.get(), texture_buffer->target(),
                       texture_buffer->name(), width(), height(), *this, plane,
                       std::move(detach));
}

GpuBuffer GpuBuffer::CopyingImageFrame(const ImageFrame& image_frame) {
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);

  auto buffer = GlTextureBuffer::Create(image_frame);

  // TODO: does this need to set the texture params? We set them again when the
  // texture is actually acccessed via GlTexture[View]. Or should they always be
  // set on creation?
  if (buffer->format() != GpuBufferFormat::kUnknown) {
    glBindTexture(GL_TEXTURE_2D, buffer->name());
    GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
        buffer->format(), /*plane=*/0, gl_context->GetGlVersion());
    gl_context->SetStandardTextureParams(buffer->target(),
                                         info.gl_internal_format);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  return GpuBuffer(std::move(buffer));
}

static void ReadTexture(const GlTextureView& view, void* output, size_t size) {
  // TODO: check buffer size? We could use glReadnPixels where available
  // (OpenGL ES 3.2, i.e. nowhere). Note that, to fully check that the read
  // won't overflow the buffer with glReadPixels, we'd also need to check or
  // reset several glPixelStore parameters (e.g. what if someone had the
  // ill-advised idea of setting GL_PACK_SKIP_PIXELS?).
  CHECK(view.gl_context());
  GlTextureInfo info =
      GlTextureInfoForGpuBufferFormat(view.gpu_buffer().format(), view.plane(),
                                      view.gl_context()->GetGlVersion());

  GLint current_fbo;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_fbo);
  CHECK_NE(current_fbo, 0);

  GLint color_attachment_name;
  glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                                        &color_attachment_name);
  if (color_attachment_name != view.name()) {
    // Save the viewport. Note that we assume that the color attachment is a
    // GL_TEXTURE_2D texture.
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    // Set the data from GLTextureView object.
    glViewport(0, 0, view.width(), view.height());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                           view.name(), 0);
    glReadPixels(0, 0, view.width(), view.height(), info.gl_format,
                 info.gl_type, output);

    // Restore from the saved viewport and color attachment name.
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           color_attachment_name, 0);
  } else {
    glReadPixels(0, 0, view.width(), view.height(), info.gl_format,
                 info.gl_type, output);
  }
}

std::unique_ptr<ImageFrame> GpuBuffer::AsImageFrame() const {
  ImageFormat::Format image_format = ImageFormatForGpuBufferFormat(format());
  auto output = absl::make_unique<ImageFrame>(
      image_format, width(), height(), ImageFrame::kGlDefaultAlignmentBoundary);
  auto view = GetGlTextureView(0, true);
  ReadTexture(view, output->MutablePixelData(), output->PixelDataSize());
  return output;
}

void GlTextureView::DoneWriting() const {
  CHECK(gpu_buffer_);
  // Inform the GlTextureBuffer that we have produced new content, and create
  // a producer sync point.
  gpu_buffer_.GetGlTextureBufferSharedPtr()->Updated(
      gl_context()->CreateSyncToken());

#ifdef __ANDROID__
  // On (some?) Android devices, the texture may need to be explicitly
  // detached from the current framebuffer.
  // TODO: is this necessary even with the unbind in BindFramebuffer?
  // It is not clear if this affected other contexts too, but let's keep it
  // while in doubt.
  GLint type = GL_NONE;
  glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE,
                                        &type);
  if (type == GL_TEXTURE) {
    GLint color_attachment = 0;
    glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                          GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME,
                                          &color_attachment);
    if (color_attachment == name()) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
  }

  // Some Android drivers log a GL_INVALID_ENUM error after the first
  // glGetFramebufferAttachmentParameteriv call if there is no bound object,
  // even though it should be ok to ask for the type and get back GL_NONE.
  // Let's just ignore any pending errors here.
  GLenum error;
  while ((error = glGetError()) != GL_NO_ERROR) {
  }

#endif  // __ANDROID__
}

#endif  // !MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe
