// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/gpu/gl_texture_buffer.h"

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"

namespace mediapipe {

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Wrap(
    GLenum target, GLuint name, int width, int height, GpuBufferFormat format,
    DeletionCallback deletion_callback) {
  return absl::make_unique<GlTextureBuffer>(target, name, width, height, format,
                                            deletion_callback);
}

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Wrap(
    GLenum target, GLuint name, int width, int height, GpuBufferFormat format,
    std::shared_ptr<GlContext> context, DeletionCallback deletion_callback) {
  return absl::make_unique<GlTextureBuffer>(target, name, width, height, format,
                                            deletion_callback, context);
}

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Create(int width, int height,
                                                         GpuBufferFormat format,
                                                         const void* data,
                                                         int alignment) {
  auto buf = absl::make_unique<GlTextureBuffer>(GL_TEXTURE_2D, 0, width, height,
                                                format, nullptr);
  if (!buf->CreateInternal(data, alignment)) {
    return nullptr;
  }
  return buf;
}

static inline int AlignedToPowerOf2(int value, int alignment) {
  // alignment must be a power of 2
  return ((value - 1) | (alignment - 1)) + 1;
}

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Create(
    const ImageFrame& image_frame) {
  int base_ws = image_frame.Width() * image_frame.NumberOfChannels() *
                image_frame.ByteDepth();
  int actual_ws = image_frame.WidthStep();
  int alignment = 0;
  std::unique_ptr<ImageFrame> temp;
  const uint8* data = image_frame.PixelData();

  // Let's see if the pixel data is tightly aligned to one of the alignments
  // supported by OpenGL, preferring 4 if possible since it's the default.
  if (actual_ws == AlignedToPowerOf2(base_ws, 4))
    alignment = 4;
  else if (actual_ws == AlignedToPowerOf2(base_ws, 1))
    alignment = 1;
  else if (actual_ws == AlignedToPowerOf2(base_ws, 2))
    alignment = 2;
  else if (actual_ws == AlignedToPowerOf2(base_ws, 8))
    alignment = 8;

  // If no GL-compatible alignment was found, we copy the data to a temporary
  // buffer, aligned to 4. We do this using another ImageFrame purely for
  // convenience.
  if (!alignment) {
    temp = std::make_unique<ImageFrame>();
    temp->CopyFrom(image_frame, 4);
    data = temp->PixelData();
    alignment = 4;
  }

  return Create(image_frame.Width(), image_frame.Height(),
                GpuBufferFormatForImageFormat(image_frame.Format()), data,
                alignment);
}

GlTextureBuffer::GlTextureBuffer(GLenum target, GLuint name, int width,
                                 int height, GpuBufferFormat format,
                                 DeletionCallback deletion_callback,
                                 std::shared_ptr<GlContext> producer_context)
    : name_(name),
      width_(width),
      height_(height),
      format_(format),
      target_(target),
      deletion_callback_(deletion_callback),
      producer_context_(producer_context) {}

bool GlTextureBuffer::CreateInternal(const void* data, int alignment) {
  auto context = GlContext::GetCurrent();
  if (!context) return false;

  producer_context_ = context;  // Save creation GL context.

  glGenTextures(1, &name_);
  if (!name_) return false;

  glBindTexture(target_, name_);
  GlTextureInfo info =
      GlTextureInfoForGpuBufferFormat(format_, 0, context->GetGlVersion());

  if (alignment != 4 && data) glPixelStorei(GL_UNPACK_ALIGNMENT, alignment);

  // See b/70294573 for details about this.
  if (info.gl_internal_format == GL_RGBA16F &&
      context->GetGlVersion() != GlVersion::kGLES2 &&
      SymbolAvailable(&glTexStorage2D)) {
    CHECK(data == nullptr) << "unimplemented";
    glTexStorage2D(target_, 1, info.gl_internal_format, width_, height_);
  } else {
    glTexImage2D(target_, 0 /* level */, info.gl_internal_format, width_,
                 height_, 0 /* border */, info.gl_format, info.gl_type, data);
  }

  if (alignment != 4 && data) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

  // TODO: does this need to set the texture params? We set them again when the
  // texture is actually acccessed via GlTexture[View]. Or should they always be
  // set on creation?
  if (format_ != GpuBufferFormat::kUnknown) {
    GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
        format_, /*plane=*/0, context->GetGlVersion());
    context->SetStandardTextureParams(target_, info.gl_internal_format);
  }

  glBindTexture(target_, 0);

  // Use the deletion callback to delete the texture on the context
  // that created it.
  CHECK(!deletion_callback_);
  deletion_callback_ = [this,
                        context](std::shared_ptr<GlSyncPoint> sync_token) {
    CHECK_NE(name_, 0);
    GLuint name_to_delete = name_;
    context->RunWithoutWaiting([name_to_delete, sync_token]() {
      // TODO: maybe we do not actually have to wait for the
      // consumer sync here. Check docs.
      sync_token->WaitOnGpu();
      DLOG_IF(ERROR, !glIsTexture(name_to_delete))
          << "Deleting invalid texture id: " << name_to_delete;
      glDeleteTextures(1, &name_to_delete);
    });
  };

  return true;
}

void GlTextureBuffer::Reuse() {
  // The old consumer sync destructor may call other contexts to delete their
  // sync fences; with a single-threaded executor, that means switching to
  // each of those contexts, grabbing its mutex. Let's do that after releasing
  // our own mutex.
  // Likewise, if we don't have sync fences and are simulating them, WaitOnGpu
  // will also require invoking the consumer context, so we should not call it
  // while holding the mutex.
  std::unique_ptr<GlMultiSyncPoint> old_consumer_sync;
  {
    absl::MutexLock lock(&consumer_sync_mutex_);
    // Reset the sync points.
    old_consumer_sync = std::move(consumer_multi_sync_);
    consumer_multi_sync_ = absl::make_unique<GlMultiSyncPoint>();
    producer_sync_ = nullptr;
  }
  old_consumer_sync->WaitOnGpu();
}

void GlTextureBuffer::Updated(std::shared_ptr<GlSyncPoint> prod_token) {
  CHECK(!producer_sync_)
      << "Updated existing texture which had not been marked for reuse!";
  producer_sync_ = std::move(prod_token);
  producer_context_ = producer_sync_->GetContext();
}

void GlTextureBuffer::DidRead(std::shared_ptr<GlSyncPoint> cons_token) const {
  absl::MutexLock lock(&consumer_sync_mutex_);
  consumer_multi_sync_->Add(std::move(cons_token));
}

GlTextureBuffer::~GlTextureBuffer() {
  if (deletion_callback_) {
    // Note: at this point there are no more consumers that could be added
    // to the consumer_multi_sync_, so it no longer needs to be protected
    // by out mutex when we hand it to the deletion callback.
    deletion_callback_(std::move(consumer_multi_sync_));
  }
}

void GlTextureBuffer::WaitUntilComplete() const {
  // Buffers created by the application (using the constructor that wraps an
  // existing texture) have no sync token and are assumed to be already
  // complete.
  if (producer_sync_) {
    producer_sync_->Wait();
  }
}

void GlTextureBuffer::WaitOnGpu() const {
  // Buffers created by the application (using the constructor that wraps an
  // existing texture) have no sync token and are assumed to be already
  // complete.
  if (producer_sync_) {
    producer_sync_->WaitOnGpu();
  }
}

void GlTextureBuffer::WaitForConsumers() {
  absl::MutexLock lock(&consumer_sync_mutex_);
  consumer_multi_sync_->Wait();
}

void GlTextureBuffer::WaitForConsumersOnGpu() {
  absl::MutexLock lock(&consumer_sync_mutex_);
  consumer_multi_sync_->WaitOnGpu();
  // TODO: should we clear the consumer_multi_sync_ here?
  // It would mean that WaitForConsumersOnGpu can be called only once, or more
  // precisely, on only one GL context.
}

GlTextureView GlTextureBuffer::GetReadView(
    internal::types<GlTextureView>, std::shared_ptr<GpuBuffer> gpu_buffer,
    int plane) const {
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);
  CHECK_EQ(plane, 0);
  // Insert wait call to sync with the producer.
  WaitOnGpu();
  GlTextureView::DetachFn detach = [this](GlTextureView& texture) {
    // Inform the GlTextureBuffer that we have finished accessing its
    // contents, and create a consumer sync point.
    DidRead(texture.gl_context()->CreateSyncToken());
  };
  return GlTextureView(gl_context.get(), target(), name(), width(), height(),
                       std::move(gpu_buffer), plane, std::move(detach),
                       nullptr);
}

GlTextureView GlTextureBuffer::GetWriteView(
    internal::types<GlTextureView>, std::shared_ptr<GpuBuffer> gpu_buffer,
    int plane) {
  auto gl_context = GlContext::GetCurrent();
  CHECK(gl_context);
  CHECK_EQ(plane, 0);
  // Insert wait call to sync with the producer.
  WaitOnGpu();
  Reuse();  // TODO: the producer wait should probably be part of Reuse in the
            // case when there are no consumers.
  GlTextureView::DoneWritingFn done_writing =
      [this](const GlTextureView& texture) { ViewDoneWriting(texture); };
  return GlTextureView(gl_context.get(), target(), name(), width(), height(),
                       std::move(gpu_buffer), plane, nullptr,
                       std::move(done_writing));
}

void GlTextureBuffer::ViewDoneWriting(const GlTextureView& view) {
  // Inform the GlTextureBuffer that we have produced new content, and create
  // a producer sync point.
  Updated(view.gl_context()->CreateSyncToken());

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

static void ReadTexture(const GlTextureView& view, GpuBufferFormat format,
                        void* output, size_t size) {
  // TODO: check buffer size? We could use glReadnPixels where available
  // (OpenGL ES 3.2, i.e. nowhere). Note that, to fully check that the read
  // won't overflow the buffer with glReadPixels, we'd also need to check or
  // reset several glPixelStore parameters (e.g. what if someone had the
  // ill-advised idea of setting GL_PACK_SKIP_PIXELS?).
  CHECK(view.gl_context());
  GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format, view.plane(), view.gl_context()->GetGlVersion());

  GLint previous_fbo;
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &previous_fbo);

  // We use a temp fbo to avoid depending on the app having an existing one.
  // TODO: keep a utility fbo around in the context?
  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                         view.name(), 0);
  glReadPixels(0, 0, view.width(), view.height(), info.gl_format, info.gl_type,
               output);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0,
                         0);
  // TODO: just set the binding to 0 to avoid the get call?
  glBindFramebuffer(GL_FRAMEBUFFER, previous_fbo);
  glDeleteFramebuffers(1, &fbo);
}

static std::shared_ptr<GpuBufferStorageImageFrame> ConvertToImageFrame(
    std::shared_ptr<GlTextureBuffer> buf) {
  ImageFormat::Format image_format =
      ImageFormatForGpuBufferFormat(buf->format());
  auto output =
      absl::make_unique<ImageFrame>(image_format, buf->width(), buf->height(),
                                    ImageFrame::kGlDefaultAlignmentBoundary);
  buf->GetProducerContext()->Run([buf, &output] {
    auto view = buf->GetReadView(internal::types<GlTextureView>{}, nullptr, 0);
    ReadTexture(view, buf->format(), output->MutablePixelData(),
                output->PixelDataSize());
  });
  return std::make_shared<GpuBufferStorageImageFrame>(std::move(output));
}

static std::shared_ptr<GlTextureBuffer> ConvertFromImageFrame(
    std::shared_ptr<GpuBufferStorageImageFrame> frame) {
  return GlTextureBuffer::Create(*frame->image_frame());
}

static auto kConverterRegistration =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<GlTextureBuffer, GpuBufferStorageImageFrame>(
            ConvertToImageFrame);
static auto kConverterRegistration2 =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<GpuBufferStorageImageFrame, GlTextureBuffer>(
            ConvertFromImageFrame);

}  // namespace mediapipe
