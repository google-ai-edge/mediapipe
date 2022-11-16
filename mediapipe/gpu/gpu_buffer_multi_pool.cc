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

#include "mediapipe/gpu/gpu_buffer_multi_pool.h"

#include <tuple>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

namespace mediapipe {

std::shared_ptr<GpuBufferMultiPool::SimplePool>
GpuBufferMultiPool::DefaultMakeSimplePool(
    const GpuBufferMultiPool::BufferSpec& spec,
    const MultiPoolOptions& options) {
  return SimplePool::Create(spec.width, spec.height, spec.format, options);
}

std::shared_ptr<GpuBufferMultiPool::SimplePool> GpuBufferMultiPool::RequestPool(
    const BufferSpec& spec) {
  std::shared_ptr<SimplePool> pool;
  std::vector<std::shared_ptr<SimplePool>> evicted;
  {
    absl::MutexLock lock(&mutex_);
    pool =
        cache_.Lookup(spec, [this](const BufferSpec& spec, int request_count) {
          return (request_count >= options_.min_requests_before_pool)
                     ? create_simple_pool_(spec, options_)
                     : nullptr;
        });
    evicted = cache_.Evict(options_.max_pool_count,
                           options_.request_count_scrub_interval);
  }
  // Evicted pools, and their buffers, will be released without holding the
  // lock.
  return pool;
}

GpuBuffer GpuBufferMultiPool::GetBuffer(int width, int height,
                                        GpuBufferFormat format) {
  BufferSpec key(width, height, format);
  std::shared_ptr<SimplePool> pool = RequestPool(key);
  if (pool) {
    // Note: we release our multipool lock before accessing the simple pool.
    return GpuBuffer(pool->GetBuffer());
  } else {
    return GpuBuffer(
        SimplePool::CreateBufferWithoutPool(width, height, format));
  }
}

}  // namespace mediapipe
