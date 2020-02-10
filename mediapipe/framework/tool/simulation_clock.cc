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

#include "mediapipe/framework/tool/simulation_clock.h"

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

SimulationClock::~SimulationClock() {
  ThreadStart();
  ThreadFinish();
}

absl::Time SimulationClock::TimeNow() {
  absl::MutexLock l(&time_mutex_);
  return time_;
}

void SimulationClock::Sleep(absl::Duration d) {
  absl::MutexLock l(&time_mutex_);
  SleepInternal(time_ + d);
}

void SimulationClock::SleepUntil(absl::Time wakeup_time) {
  absl::MutexLock l(&time_mutex_);
  SleepInternal(wakeup_time);
}

void SimulationClock::SleepInternal(absl::Time wakeup_time) {
  Waiter waiter;
  waiters_.insert({wakeup_time, &waiter});
  num_running_--;
  TryAdvanceTime();
  while (waiter.sleeping) {
    waiter.cond.Wait(&time_mutex_);
  }
  num_running_++;
}

void SimulationClock::ThreadStart() {
  absl::MutexLock l(&time_mutex_);
  num_running_++;
}

void SimulationClock::ThreadFinish() {
  absl::MutexLock l(&time_mutex_);
  num_running_--;
  TryAdvanceTime();
}

void SimulationClock::TryAdvanceTime() {
  if (num_running_ == 0 && !waiters_.empty()) {
    VLOG(2) << "Advance time from: " << absl::ToUnixMicros(time_)
            << " to: " << absl::ToUnixMicros(waiters_.begin()->first);
    time_ = waiters_.begin()->first;
    Waiter* waiter = waiters_.begin()->second;
    waiters_.erase(waiters_.begin());
    waiter->sleeping = false;
    waiter->cond.Signal();
  }
}

}  // namespace mediapipe
