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

#include "mediapipe/util/tracking/parallel_invoker.h"

// Choose between ThreadPool, OpenMP and serial execution.
// Note only one parallel_using_* directive can be active.
int flags_parallel_invoker_mode = PARALLEL_INVOKER_MAX_VALUE;
int flags_parallel_invoker_max_threads = 4;

namespace mediapipe {

#if defined(PARALLEL_INVOKER_ACTIVE)
ThreadPool* ParallelInvokerThreadPool() {
  static ThreadPool* pool = []() -> ThreadPool* {
    ThreadPool* new_pool =
        new ThreadPool("ParallelInvoker", flags_parallel_invoker_max_threads);
    new_pool->StartWorkers();
    return new_pool;
  }();
  return pool;
}
#endif

}  // namespace mediapipe
