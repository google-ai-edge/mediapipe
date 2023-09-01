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

#include "mediapipe/framework/deps/monotonic_clock.h"

#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

namespace mediapipe {

// This state, which contains the "guts" of MonotonicClockImpl, is separate
// from the class instance so that it can be shared to implement a
// SynchronizedMonotonicClock.  (The per-instance state of MonotonicClock is
// just for frills like the correction metrics and callback.)  It lives in this
// private namespace so that test code can use it without exposing it to the
// world.
struct MonotonicClock::State {
  // The clock whose time is being corrected.
  Clock* raw_clock;
  absl::Mutex lock;
  // The largest time ever returned by Now().
  absl::Time max_time ABSL_GUARDED_BY(lock);
  explicit State(Clock* clock)
      : raw_clock(clock), max_time(absl::UnixEpoch()) {}
};

using State = MonotonicClock::State;

class MonotonicClockImpl : public MonotonicClock {
 public:
  // By default, MonotonicClockImpl owns the state_.  ReleaseState(), below,
  // can be used to prevent the MCI destructor from deleting a shared state_.
  explicit MonotonicClockImpl(State* state)
      : state_(state),
        state_owned_(true),
        last_raw_time_(absl::UnixEpoch()),
        correction_count_(0),
        max_correction_(absl::ZeroDuration()) {}

  MonotonicClockImpl(const MonotonicClockImpl&) = delete;
  MonotonicClockImpl& operator=(const MonotonicClockImpl&) = delete;

  virtual ~MonotonicClockImpl() {
    if (state_owned_) delete state_;
  }

  // Absolve this object of responsibility for state_.
  void ReleaseState() {
    ABSL_CHECK(state_owned_);
    state_owned_ = false;
  }

  //
  // The Clock interface (see util/time/clock.h).
  //

  // The logic in TimeNow() is based on GFS_NowMS().
  virtual absl::Time TimeNow() {
    // These variables save some state from the critical section below.
    absl::Time raw_time;
    absl::Time local_max_time;
    absl::Time local_last_raw_time;

    // As there are several early exits from this function, use absl::MutexLock.
    {
      absl::MutexLock m(&state_->lock);

      // Check consistency of internal data with state_.
      ABSL_CHECK_LE(last_raw_time_, state_->max_time)
          << "non-monotonic behavior: last_raw_time_=" << last_raw_time_
          << ", max_time=" << state_->max_time;

      raw_time = state_->raw_clock->TimeNow();

      // Normal case: time is advancing.  Update state and return the raw time.
      if (raw_time >= state_->max_time) {
        last_raw_time_ = raw_time;
        state_->max_time = raw_time;
        return raw_time;
      }

      // Exceptional case: Raw time is within a window of a previous backward
      // jump.  We do not run any callbacks or update metrics here since we
      // already did that when the backward jump was detected.
      if (raw_time >= last_raw_time_) {
        last_raw_time_ = raw_time;
        return state_->max_time;
      }

      // Exceptional case: Raw time jumped backward.  Remainder of function
      // handles this case.
      //
      // First, update correction metrics.
      ++correction_count_;
      absl::Duration delta = state_->max_time - raw_time;
      ABSL_CHECK_LT(absl::ZeroDuration(), delta);
      if (delta > max_correction_) {
        max_correction_ = delta;
      }

      // Copy state into local vars before updating last_raw_time_ and leaving
      // the critical section.
      local_max_time = state_->max_time;
      local_last_raw_time = last_raw_time_;
      last_raw_time_ = raw_time;
    }  // absl::MutexLock

    // Return the saved maximum time.
    return local_max_time;
  }

  // The strategy of Sleep and SleepUntil is K.I.S.S.: set an alarm on the
  // raw_clock for the desired wakeup_time, and then snooze the alarm if we wake
  // up too soon.  This guarantees that the caller won't wake up too soon (which
  // would require us to advance monotonic time simply by the act of waking up),
  // however the caller may sleep for much longer (in monotonic time) if
  // monotonic time jumps far into the future.  Whether or not this happens
  // depends on the behavior of the raw clock.
  virtual void Sleep(absl::Duration d) {
    absl::Time wakeup_time = TimeNow() + d;
    SleepUntil(wakeup_time);
  }

  virtual void SleepUntil(absl::Time wakeup_time) {
    while (TimeNow() < wakeup_time) {
      state_->raw_clock->SleepUntil(wakeup_time);
    }
  }

  //
  // End of Clock interface.
  //

 private:
  // Get metrics about time corrections.
  virtual void GetCorrectionMetrics(int* correction_count,
                                    double* max_correction) {
    absl::MutexLock l(&state_->lock);
    if (correction_count != nullptr) *correction_count = correction_count_;
    if (max_correction != nullptr)
      *max_correction = absl::FDivDuration(max_correction_, absl::Seconds(1));
  }

  // Reset values returned by GetCorrectionMetrics().
  virtual void ResetCorrectionMetrics() {
    absl::MutexLock l(&state_->lock);
    correction_count_ = 0;
    max_correction_ = absl::ZeroDuration();
  }

  // The guts of the monotonic clock.  Caution: this may point to a static
  // object.
  State* state_;
  // If true, this object owns state_ and is responsible for deallocating it.
  bool state_owned_;

  // last_raw_time_ remembers the last value obtained from raw_clock_.
  // It prevents spurious calls to ReportCorrection when time moves
  // forward by a smaller amount than a prior backward jump.
  absl::Time last_raw_time_ ABSL_GUARDED_BY(state_->lock);

  // Variables that keep track of time corrections made by this instance of
  // MonotonicClock.  (All such metrics are instance-local for reasons
  // described earlier.)
  int correction_count_ ABSL_GUARDED_BY(state_->lock);
  absl::Duration max_correction_ ABSL_GUARDED_BY(state_->lock);
};

// Factory methods.
MonotonicClock* MonotonicClock::CreateMonotonicClock(Clock* clock) {
  State* state = new State(clock);
  // MonotonicClockImpl takes ownership of state.
  return new MonotonicClockImpl(state);
}

namespace {
State* GlobalSyncState() {
  static State* sync_state = new State(Clock::RealClock());
  return sync_state;
}
}  // namespace

// The reason that SynchronizedMonotonicClock is not implemented as a singleton
// is so that different code bases can handle clock corrections their own way.
MonotonicClock* MonotonicClock::CreateSynchronizedMonotonicClock() {
  MonotonicClockImpl* clock = new MonotonicClockImpl(GlobalSyncState());
  // Release ownership of sync_state.
  clock->ReleaseState();
  return clock;
}

// Test access methods.
void MonotonicClockAccess::SynchronizedMonotonicClockReset() {
  ABSL_LOG(INFO) << "Resetting SynchronizedMonotonicClock";
  State* sync_state = GlobalSyncState();
  absl::MutexLock m(&sync_state->lock);
  sync_state->max_time = absl::UnixEpoch();
}

State* MonotonicClockAccess::CreateMonotonicClockState(Clock* raw_clock) {
  return new State(raw_clock);
}

void MonotonicClockAccess::DeleteMonotonicClockState(State* state) {
  delete state;
}

MonotonicClock* MonotonicClockAccess::CreateMonotonicClock(State* state) {
  MonotonicClockImpl* clock = new MonotonicClockImpl(state);
  // Release ownership of sync_state.
  clock->ReleaseState();
  return clock;
}

}  // namespace mediapipe
