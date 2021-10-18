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

#ifndef MEDIAPIPE_GPU_GL_TEXTURE_VIEW_H_
#define MEDIAPIPE_GPU_GL_TEXTURE_VIEW_H_

#include <functional>
#include <memory>
#include <utility>

#include "mediapipe/gpu/gl_base.h"

namespace mediapipe {

class GlContext;
class GlTextureViewManager;
class GpuBuffer;

class GlTextureView {
 public:
  GlTextureView() {}
  ~GlTextureView() { Release(); }
  // TODO: make this class move-only.

  GlContext* gl_context() const { return gl_context_; }
  int width() const { return width_; }
  int height() const { return height_; }
  GLenum target() const { return target_; }
  GLuint name() const { return name_; }
  const GpuBuffer& gpu_buffer() const { return *gpu_buffer_; }
  int plane() const { return plane_; }

  using DetachFn = std::function<void(GlTextureView&)>;
  using DoneWritingFn = std::function<void(const GlTextureView&)>;

 private:
  friend class GpuBuffer;
  friend class GlTextureBuffer;
  friend class GpuBufferStorageCvPixelBuffer;
  GlTextureView(GlContext* context, GLenum target, GLuint name, int width,
                int height, std::shared_ptr<GpuBuffer> gpu_buffer, int plane,
                DetachFn detach, DoneWritingFn done_writing)
      : gl_context_(context),
        target_(target),
        name_(name),
        width_(width),
        height_(height),
        gpu_buffer_(std::move(gpu_buffer)),
        plane_(plane),
        detach_(std::move(detach)),
        done_writing_(std::move(done_writing)) {}

  // TODO: remove this friend declaration.
  friend class GlTexture;
  void Release();
  // TODO: make this non-const.
  void DoneWriting() const {
    if (done_writing_) done_writing_(*this);
  }

  GlContext* gl_context_ = nullptr;
  GLenum target_ = GL_TEXTURE_2D;
  GLuint name_ = 0;
  // Note: when scale is not 1, we still give the nominal size of the image.
  int width_ = 0;
  int height_ = 0;
  std::shared_ptr<GpuBuffer> gpu_buffer_;  // using shared_ptr temporarily
  int plane_ = 0;
  DetachFn detach_;
  DoneWritingFn done_writing_;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GL_TEXTURE_VIEW_H_
