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

// This class lets calculators allocate GpuBuffers of various sizes, caching
// and reusing them as needed. It does so by automatically creating and using
// platform-specific buffer pools for the requested sizes.
//
// This class is not meant to be used directly by calculators, but is instead
// used by GlCalculatorHelper to allocate buffers.

#ifndef MEDIAPIPE_GPU_MULTI_POOL_H_
#define MEDIAPIPE_GPU_MULTI_POOL_H_

namespace mediapipe {

struct MultiPoolOptions {
  // Keep this many buffers allocated for a given frame size.
  int keep_count = 2;
  // The maximum size of the GpuBufferMultiPool. When the limit is reached, the
  // oldest BufferSpec will be dropped.
  int max_pool_count = 10;
  // Time in seconds after which an inactive buffer can be dropped from the
  // pool. Currently only used with CVPixelBufferPool.
  float max_inactive_buffer_age = 0.25;
  // Skip allocating a buffer pool until at least this many requests have been
  // made for a given BufferSpec.
  int min_requests_before_pool = 2;
  // Do a deeper flush every this many requests.
  int request_count_scrub_interval = 50;
};

static constexpr MultiPoolOptions kDefaultMultiPoolOptions;

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_MULTI_POOL_H_
