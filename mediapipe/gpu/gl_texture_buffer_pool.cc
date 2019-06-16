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

#include "mediapipe/gpu/gl_texture_buffer_pool.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {

GlTextureBufferPool::GlTextureBufferPool(int width, int height,
                                         GpuBufferFormat format, int keep_count)
    : width_(width),
      height_(height),
      format_(format),
      keep_count_(keep_count) {}

GlTextureBufferSharedPtr GlTextureBufferPool::GetBuffer() {
  absl::MutexLock lock(&mutex_);

  std::unique_ptr<GlTextureBuffer> buffer;
  if (available_.empty()) {
    buffer = GlTextureBuffer::Create(width_, height_, format_);
    if (!buffer) return nullptr;
  } else {
    buffer = std::move(available_.back());
    available_.pop_back();
    buffer->Reuse();
  }

  ++in_use_count_;

  // Return a shared_ptr with a custom deleter that adds the buffer back
  // to our available list.
  std::weak_ptr<GlTextureBufferPool> weak_pool(shared_from_this());
  return std::shared_ptr<GlTextureBuffer>(buffer.release(),
                                          [weak_pool](GlTextureBuffer* buf) {
                                            auto pool = weak_pool.lock();
                                            if (pool) {
                                              pool->Return(buf);
                                            } else {
                                              delete buf;
                                            }
                                          });
}

std::pair<int, int> GlTextureBufferPool::GetInUseAndAvailableCounts() {
  absl::MutexLock lock(&mutex_);
  return {in_use_count_, available_.size()};
}

void GlTextureBufferPool::Return(GlTextureBuffer* buf) {
  absl::MutexLock lock(&mutex_);
  --in_use_count_;
  available_.emplace_back(buf);
  TrimAvailable();
}

void GlTextureBufferPool::TrimAvailable() {
  int keep = std::max(keep_count_ - in_use_count_, 0);
  if (available_.size() > keep) {
    available_.resize(keep);
  }
}

}  // namespace mediapipe
