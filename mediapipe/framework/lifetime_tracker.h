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

#ifndef MEDIAPIPE_FRAMEWORK_LIFETIME_TRACKER_H_
#define MEDIAPIPE_FRAMEWORK_LIFETIME_TRACKER_H_

#include <atomic>

#include "absl/memory/memory.h"

namespace mediapipe {

// This class can be used to create objects whose lifetime is tracked by a
// counter. This is useful for testing.
// There is a separate counter per LifetimeTracker instance, and it counts the
// number of LifetimeTracker::Object instances created by that tracker.
// Therefore, you can use a single tracker with multiple objects to track
// overall behavior; or you can use separate trackers, with one object each, if
// you need to track each object's lifetime separately.
class LifetimeTracker {
 public:
  class Object {
   public:
    explicit Object(LifetimeTracker* tracker) : tracker_(tracker) {
      ++tracker_->live_count_;
    }
    ~Object() { --tracker_->live_count_; }

   private:
    LifetimeTracker* const tracker_;
  };

  // Creates and returns a new tracked object.
  std::unique_ptr<Object> MakeObject() {
    return absl::make_unique<Object>(this);
  }

  // Returns the number of tracked objects currently alive.
  int live_count() { return live_count_; }

 private:
  std::atomic<int> live_count_ = ATOMIC_VAR_INIT(0);
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_LIFETIME_TRACKER_H_
