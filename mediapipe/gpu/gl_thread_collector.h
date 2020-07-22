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

#ifndef MEDIAPIPE_GPU_GL_THREAD_COLLECTOR_H_
#define MEDIAPIPE_GPU_GL_THREAD_COLLECTOR_H_

#include <cstdlib>

#if defined(MEDIAPIPE_USING_SWIFTSHADER) && !defined(NDEBUG)
#define MEDIAPIPE_NEEDS_GL_THREAD_COLLECTOR 1
#endif

#if MEDIAPIPE_NEEDS_GL_THREAD_COLLECTOR
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/deps/no_destructor.h"
#endif  // MEDIAPIPE_NEEDS_GL_THREAD_COLLECTOR

namespace mediapipe {

#if MEDIAPIPE_NEEDS_GL_THREAD_COLLECTOR

class GlThreadCollector {
 public:
  static void ThreadStarting() { Collector().ChangeCount(1); }

  static void ThreadEnding() { Collector().ChangeCount(-1); }

 private:
  GlThreadCollector() { std::atexit(WaitForThreadsToTerminate); }

  static GlThreadCollector& Collector() {
    static NoDestructor<GlThreadCollector> collector;
    return *collector;
  }

  static void WaitForThreadsToTerminate() { Collector().Wait(); }

  void ChangeCount(int delta) {
    absl::MutexLock l(&mutex_);
    active_threads_ += delta;
  }

  void Wait() {
    auto done = [this]() {
      mutex_.AssertReaderHeld();
      return active_threads_ == 0;
    };
    absl::MutexLock l(&mutex_);
    mutex_.Await(absl::Condition(&done));
  }

  absl::Mutex mutex_;
  int active_threads_ ABSL_GUARDED_BY(mutex_) = 0;
  friend NoDestructor<GlThreadCollector>;
};
#else
class GlThreadCollector {
 public:
  static void ThreadStarting() {}
  static void ThreadEnding() {}
};
#endif  // MEDIAPIPE_NEEDS_GL_THREAD_COLLECTOR

}  // namespace mediapipe

#endif  // MEDIAPIPE_GPU_GL_THREAD_COLLECTOR_H_
