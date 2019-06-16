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

#include "mediapipe/framework/deps/clock.h"

#include "absl/time/clock.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

namespace {

// -----------------------------------------------------------------
// RealTimeClock
//
// This class is thread-safe.
class RealTimeClock : public Clock {
 public:
  virtual ~RealTimeClock() {
    LOG(FATAL) << "RealTimeClock should never be destroyed";
  }

  absl::Time TimeNow() override { return absl::Now(); }

  void Sleep(absl::Duration d) override { absl::SleepFor(d); }

  void SleepUntil(absl::Time wakeup_time) override {
    absl::Duration d = wakeup_time - TimeNow();
    if (d > absl::ZeroDuration()) {
      Sleep(d);
    }
  }
};

}  // namespace

Clock::~Clock() {}

Clock* Clock::RealClock() {
  static RealTimeClock* rtclock = new RealTimeClock;
  return rtclock;
}

}  // namespace mediapipe
