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

#include "mediapipe/framework/formats/image_frame_pool.h"

#include "absl/synchronization/mutex.h"

namespace mediapipe {

ImageFramePool::ImageFramePool(int width, int height,
                               ImageFormat::Format format, int keep_count)
    : width_(width),
      height_(height),
      format_(format),
      keep_count_(keep_count) {}

ImageFrameSharedPtr ImageFramePool::GetBuffer() {
  std::unique_ptr<ImageFrame> buffer;

  {
    absl::MutexLock lock(&mutex_);
    if (available_.empty()) {
      // Fix alignment at 4 for best compatability with OpenGL.
      buffer = std::make_unique<ImageFrame>(
          format_, width_, height_, ImageFrame::kGlDefaultAlignmentBoundary);
      if (!buffer) return nullptr;
    } else {
      buffer = std::move(available_.back());
      available_.pop_back();
    }

    ++in_use_count_;
  }

  // Return a shared_ptr with a custom deleter that adds the buffer back
  // to our available list.
  std::weak_ptr<ImageFramePool> weak_pool(shared_from_this());
  return std::shared_ptr<ImageFrame>(buffer.release(),
                                     [weak_pool](ImageFrame* buf) {
                                       auto pool = weak_pool.lock();
                                       if (pool) {
                                         pool->Return(buf);
                                       } else {
                                         delete buf;
                                       }
                                     });
}

std::pair<int, int> ImageFramePool::GetInUseAndAvailableCounts() {
  absl::MutexLock lock(&mutex_);
  return {in_use_count_, available_.size()};
}

void ImageFramePool::Return(ImageFrame* buf) {
  std::vector<std::unique_ptr<ImageFrame>> trimmed;
  {
    absl::MutexLock lock(&mutex_);
    --in_use_count_;
    available_.emplace_back(buf);
    TrimAvailable(&trimmed);
  }
  // The trimmed buffers will be released without holding the lock.
}

void ImageFramePool::TrimAvailable(
    std::vector<std::unique_ptr<ImageFrame>>* trimmed) {
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
