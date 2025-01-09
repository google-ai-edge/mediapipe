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

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_view.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_buffer_storage.h"
#include "mediapipe/gpu/gpu_buffer_storage_image_frame.h"

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER
#include "mediapipe/gpu/gl_texture_util.h"
#include "mediapipe/gpu/gpu_buffer_storage_cv_pixel_buffer.h"
#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

namespace mediapipe {

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Wrap(
    GLenum target, GLuint name, int width, int height, GpuBufferFormat format,
    DeletionCallback deletion_callback) {
  return std::make_unique<GlTextureBuffer>(target, name, width, height, format,
                                           deletion_callback);
}

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Wrap(
    GLenum target, GLuint name, int width, int height, GpuBufferFormat format,
    std::shared_ptr<GlContext> context, DeletionCallback deletion_callback) {
  return std::make_unique<GlTextureBuffer>(target, name, width, height, format,
                                           deletion_callback, context);
}

std::unique_ptr<GlTextureBuffer> GlTextureBuffer::Create(int width, int height,
                                                         GpuBufferFormat format,
                                                         const void* data,
                                                         int alignment) {
  auto buf = std::make_unique<GlTextureBuffer>(GL_TEXTURE_2D, 0, width, height,
                                               format, nullptr);
  if (!buf->CreateInternal(data, alignment)) {
    ABSL_LOG(WARNING) << absl::StrFormat(
        "Failed to create a GL texture: %d x %d, %d", width, height, format);
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
  const uint8_t* data = image_frame.PixelData();

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
  if (!context) {
    ABSL_LOG(WARNING) << "Cannot create a GL texture without a valid context";
    return false;
  }

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
    ABSL_CHECK(data == nullptr) << "unimplemented";
    glTexStorage2D(target_, 1, info.gl_internal_format, width_, height_);
  } else if (info.immutable) {
    ABSL_CHECK(SymbolAvailable(&glTexStorage2D) &&
               context->GetGlVersion() != GlVersion::kGLES2)
        << "Immutable GpuBuffer format requested is not supported in this "
        << "GlContext. Format was " << static_cast<uint32_t>(format_);
    ABSL_CHECK(data == nullptr) << "unimplemented";
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
  ABSL_CHECK(!deletion_callback_);
  deletion_callback_ = [this,
                        context](std::shared_ptr<GlSyncPoint> sync_token) {
    ABSL_CHECK_NE(name_, 0);
    GLuint name_to_delete = name_;
    context->RunWithoutWaiting([name_to_delete]() {
      // Note that we do not wait for consumers to be done before deleting the
      // texture. Based on a reading of the GLES 3.0 spec, appendix D:
      // - when a texture is deleted, it is _not_ automatically unbound from
      //   bind points in other contexts;
      // - when a texture is deleted, its name becomes immediately invalid, but
      //   the actual object is not deleted until it is no longer in use, i.e.
      //   attached to a container object or bound to a context;
      // - deleting an object is not an operation that changes its contents;
      // - within each context, commands are executed sequentially, so it seems
      //   like an unbind that follows a command that reads a texture should not
      //   take effect until the GPU has actually finished executing the
      //   previous commands.
      // The final point is the least explicit in the docs, but it is implied by
      // normal single-context behavior. E.g. if you do bind, delete, render,
      // unbind, the object is not deleted until the unbind, and it waits for
      // the render to finish.
      ABSL_DLOG_IF(ERROR, !glIsTexture(name_to_delete))
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
    consumer_multi_sync_ = std::make_unique<GlMultiSyncPoint>();
    producer_sync_ = nullptr;
  }
  old_consumer_sync->WaitOnGpu();
}

void GlTextureBuffer::Updated(std::shared_ptr<GlSyncPoint> prod_token) {
  ABSL_CHECK(!producer_sync_)
      << "Updated existing texture which had not been marked for reuse!";
  ABSL_CHECK(prod_token);
  producer_sync_ = std::move(prod_token);
  const auto& synced_context = producer_sync_->GetContext();
  if (synced_context) {
    producer_context_ = synced_context;
  }
}

void GlTextureBuffer::DidRead(std::shared_ptr<GlSyncPoint> cons_token) const {
  absl::MutexLock lock(&consumer_sync_mutex_);
  if (cons_token) {
    consumer_multi_sync_->Add(std::move(cons_token));
  } else {
    // TODO: change to a CHECK.
    ABSL_LOG_FIRST_N(WARNING, 5) << "unexpected null sync in DidRead";
  }
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

GlTextureView GlTextureBuffer::GetReadView(internal::types<GlTextureView>,
                                           int plane) const {
  auto gl_context = GlContext::GetCurrent();
  ABSL_CHECK(gl_context);
  ABSL_CHECK_EQ(plane, 0);
  // Note that this method is only supposed to be called by GpuBuffer, which
  // ensures this condition is satisfied.
  ABSL_DCHECK(!weak_from_this().expired())
      << "GlTextureBuffer must be held in shared_ptr to get a GlTextureView";
  // Insert wait call to sync with the producer.
  WaitOnGpu();
  GlTextureView::DetachFn detach =
      [texbuf = shared_from_this()](GlTextureView& texture) {
        // Inform the GlTextureBuffer that we have finished accessing its
        // contents, and create a consumer sync point.
        texbuf->DidRead(texture.gl_context()->CreateSyncToken());
      };
  return GlTextureView(gl_context.get(), target(), name(), width(), height(),
                       plane, std::move(detach), nullptr);
}

GlTextureView GlTextureBuffer::GetWriteView(internal::types<GlTextureView>,
                                            int plane) {
  auto gl_context = GlContext::GetCurrent();
  ABSL_CHECK(gl_context);
  ABSL_CHECK_EQ(plane, 0);
  // Note that this method is only supposed to be called by GpuBuffer, which
  // ensures this condition is satisfied.
  ABSL_DCHECK(!weak_from_this().expired())
      << "GlTextureBuffer must be held in shared_ptr to get a GlTextureView";
  // Insert wait call to sync with the producer.
  WaitOnGpu();
  Reuse();  // TODO: the producer wait should probably be part of Reuse in the
            // case when there are no consumers.
  GlTextureView::DoneWritingFn done_writing =
      [texbuf = shared_from_this()](const GlTextureView& texture) {
        texbuf->ViewDoneWriting(texture);
      };
  return GlTextureView(gl_context.get(), target(), name(), width(), height(),
                       plane, nullptr, std::move(done_writing));
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

static void ReadTexture(GlContext& ctx, const GlTextureView& view,
                        GpuBufferFormat format, void* output, size_t size) {
  // TODO: check buffer size? We could use glReadnPixels where available
  // (OpenGL ES 3.2, i.e. nowhere). Note that, to fully check that the read
  // won't overflow the buffer with glReadPixels, we'd also need to check or
  // reset several glPixelStore parameters (e.g. what if someone had the
  // ill-advised idea of setting GL_PACK_SKIP_PIXELS?).
  ABSL_CHECK(view.gl_context());
  GlTextureInfo info = GlTextureInfoForGpuBufferFormat(
      format, view.plane(), view.gl_context()->GetGlVersion());

  GLuint fbo = kUtilityFramebuffer.Get(ctx);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, view.target(),
                         view.name(), 0);
  glReadPixels(0, 0, view.width(), view.height(), info.gl_format, info.gl_type,
               output);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0,
                         0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static std::shared_ptr<GpuBufferStorageImageFrame> ConvertToImageFrame(
    std::shared_ptr<GlTextureBuffer> buf) {
  ImageFormat::Format image_format =
      ImageFormatForGpuBufferFormat(buf->format());
  auto output =
      std::make_unique<ImageFrame>(image_format, buf->width(), buf->height(),
                                   ImageFrame::kGlDefaultAlignmentBoundary);
  auto ctx = GlContext::GetCurrent();
  if (!ctx) ctx = buf->GetProducerContext();
  ctx->Run([buf, &output, &ctx] {
    auto view = buf->GetReadView(internal::types<GlTextureView>{}, /*plane=*/0);
    ReadTexture(*ctx, view, buf->format(), output->MutablePixelData(),
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

#if MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

static std::shared_ptr<GpuBufferStorageCvPixelBuffer> ConvertToCvPixelBuffer(
    std::shared_ptr<GlTextureBuffer> buf) {
  auto output = std::make_unique<GpuBufferStorageCvPixelBuffer>(
      buf->width(), buf->height(), buf->format());
  auto ctx = GlContext::GetCurrent();
  if (!ctx) ctx = buf->GetProducerContext();
  ctx->Run([buf, &output] {
    TempGlFramebuffer framebuffer;
    auto src = buf->GetReadView(internal::types<GlTextureView>{}, /*plane=*/0);
    auto dst =
        output->GetWriteView(internal::types<GlTextureView>{}, /*plane=*/0);
    CopyGlTexture(src, dst);
    glFlush();
  });
  return output;
}

static auto kConverterRegistrationCvpb =
    internal::GpuBufferStorageRegistry::Get()
        .RegisterConverter<GlTextureBuffer, GpuBufferStorageCvPixelBuffer>(
            ConvertToCvPixelBuffer);

#endif  // MEDIAPIPE_GPU_BUFFER_USE_CV_PIXEL_BUFFER

}  // namespace mediapipe
