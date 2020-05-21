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
  std::unique_ptr<GlTextureBuffer> buffer;
  bool reuse = false;

  {
    absl::MutexLock lock(&mutex_);
    if (available_.empty()) {
      buffer = GlTextureBuffer::Create(width_, height_, format_);
      if (!buffer) return nullptr;
    } else {
      buffer = std::move(available_.back());
      available_.pop_back();
      reuse = true;
    }

    ++in_use_count_;
  }

  // This needs to wait on consumer sync points, therefore it should not be
  // done while holding the mutex.
  if (reuse) {
    buffer->Reuse();
  }

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
  std::vector<std::unique_ptr<GlTextureBuffer>> trimmed;
  {
    absl::MutexLock lock(&mutex_);
    --in_use_count_;
    available_.emplace_back(buf);
    TrimAvailable(&trimmed);
  }
  // The trimmed buffers will be released without holding the lock.
}

void GlTextureBufferPool::TrimAvailable(
    std::vector<std::unique_ptr<GlTextureBuffer>>* trimmed) {
  int keep = std::max(keep_count_ - in_use_count_, 0);
  if (available_.size() > keep) {
    auto trim_it = std::next(available_.begin(), keep);
    if (trimmed) {
      std::move(trim_it, available_.end(), std::back_inserter(*trimmed));
    }
    available_.erase(trim_it, available_.end());
  }
}

}  // namespace mediapipe
