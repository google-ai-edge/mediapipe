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
                                                         const void* data) {
  auto buf = absl::make_unique<GlTextureBuffer>(GL_TEXTURE_2D, 0, width, height,
                                                format, nullptr);
  if (!buf->CreateInternal(data)) {
    return nullptr;
  }
  return buf;
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

bool GlTextureBuffer::CreateInternal(const void* data) {
  auto context = GlContext::GetCurrent();
  if (!context) return false;

  producer_context_ = context;  // Save creation GL context.

  glGenTextures(1, &name_);
  if (!name_) return false;

  glBindTexture(target_, name_);
  GlTextureInfo info = GlTextureInfoForGpuBufferFormat(format_, 0);

  // See b/70294573 for details about this.
  if (info.gl_internal_format == GL_RGBA16F &&
      SymbolAvailable(&glTexStorage2D)) {
    CHECK(data == nullptr) << "unimplemented";
    glTexStorage2D(target_, 1, info.gl_internal_format, width_, height_);
  } else {
    glTexImage2D(target_, 0 /* level */, info.gl_internal_format, width_,
                 height_, 0 /* border */, info.gl_format, info.gl_type, data);
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
  std::unique_ptr<GlMultiSyncPoint> old_consumer_sync;
  {
    absl::MutexLock lock(&consumer_sync_mutex_);
    consumer_multi_sync_->WaitOnGpu();
    // Reset the sync points.
    old_consumer_sync = std::move(consumer_multi_sync_);
    consumer_multi_sync_ = absl::make_unique<GlMultiSyncPoint>();
    producer_sync_ = nullptr;
  }
}

void GlTextureBuffer::Updated(std::shared_ptr<GlSyncPoint> prod_token) {
  CHECK(!producer_sync_)
      << "Updated existing texture which had not been marked for reuse!";
  producer_sync_ = std::move(prod_token);
  producer_context_ = producer_sync_->GetContext();
}

void GlTextureBuffer::DidRead(std::shared_ptr<GlSyncPoint> cons_token) {
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

void GlTextureBuffer::WaitUntilComplete() {
  // Buffers created by the application (using the constructor that wraps an
  // existing texture) have no sync token and are assumed to be already
  // complete.
  if (producer_sync_) {
    producer_sync_->Wait();
  }
}

void GlTextureBuffer::WaitOnGpu() {
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

}  // namespace mediapipe
