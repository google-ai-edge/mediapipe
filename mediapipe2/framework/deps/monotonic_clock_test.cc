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

#include <stddef.h>

#include <memory>
#include <random>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/integral_types.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/threadpool.h"
#include "mediapipe/framework/tool/simulation_clock.h"

namespace mediapipe {

using RandomEngine = std::mt19937_64;
using State = MonotonicClock::State;

// absl::Now() recomputes clock drift approx. every 2 seconds, so run real
// clock tests for at least that long.
static const absl::Duration kDefaultRealTest = absl::Seconds(2.5);

class MonotonicClockTest : public testing::Test {
 protected:
  MonotonicClockTest() {}
  virtual ~MonotonicClockTest() {}

  void SetUp() override {
    MonotonicClockAccess::SynchronizedMonotonicClockReset();
  }

  void VerifyCorrectionMetrics(MonotonicClock* clock,
                               int num_corrections_expect,
                               double max_correction_expect) {
    int clock_num_corrections;
    double clock_max_correction;
    clock->GetCorrectionMetrics(&clock_num_corrections, &clock_max_correction);
    ASSERT_EQ(num_corrections_expect, clock_num_corrections);
    ASSERT_DOUBLE_EQ(max_correction_expect, clock_max_correction);
  }

  // This test produces no time corrections.
  void TestSimulatedForwardTime(SimulationClock* sim_clock,
                                MonotonicClock* mono_clock) {
    absl::Time base_time = sim_clock->TimeNow();
    ASSERT_EQ(base_time, mono_clock->TimeNow());
    sim_clock->Sleep(absl::Seconds(10));
    ASSERT_EQ(base_time + absl::Seconds(10), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(10), mono_clock->TimeNow());
    sim_clock->Sleep(absl::Seconds(10));
    ASSERT_EQ(base_time + absl::Seconds(20), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(20), mono_clock->TimeNow());
    sim_clock->Sleep(absl::Seconds(5));
    ASSERT_EQ(base_time + absl::Seconds(25), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(25), mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 0, 0.0);
  }

  // This test produces three corrections: one with arguments
  // (50, 100, 100), one with (80, 90, 100), and one with (60, 105, 105).
  void TestSimulatedBackwardTime(SimulationClock* sim_clock,
                                 MonotonicClock* mono_clock) {
    absl::Time base_time = sim_clock->TimeNow();
    sim_clock->Sleep(absl::Seconds(100));
    ASSERT_EQ(base_time + absl::Seconds(100), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(100), mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 0, 0.0);
    // Time moves backward -- expect a correction.
    sim_clock->Sleep(absl::Seconds(-50));
    ASSERT_EQ(base_time + absl::Seconds(50), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(100),  // correction
              mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 1, 50.0);
    // Time moves forward, but not enough to exceed the last value returned by
    // TimeNow(). No correction in this case.
    sim_clock->Sleep(absl::Seconds(20));
    ASSERT_EQ(base_time + absl::Seconds(70), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(100), mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 1, 50.0);
    sim_clock->Sleep(absl::Seconds(20));
    ASSERT_EQ(base_time + absl::Seconds(90), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(100), mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 1, 50.0);
    // Time moves backwards again -- expect a correction.
    sim_clock->Sleep(absl::Seconds(-10));
    ASSERT_EQ(base_time + absl::Seconds(80), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(100),  // correction
              mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 2, 50.0);
    // Time moves forward enough to advance monotonic time.
    sim_clock->Sleep(absl::Seconds(25));
    ASSERT_EQ(base_time + absl::Seconds(105), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(105), mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 2, 50.0);
    // Time moves backward again.
    sim_clock->Sleep(absl::Seconds(-45));
    ASSERT_EQ(base_time + absl::Seconds(60), sim_clock->TimeNow());
    ASSERT_EQ(base_time + absl::Seconds(105),  // correction
              mono_clock->TimeNow());
    VerifyCorrectionMetrics(mono_clock, 3, 50.0);

    // Reset metrics and re-verify.
    mono_clock->ResetCorrectionMetrics();
    VerifyCorrectionMetrics(mono_clock, 0, 0.0);
  }

  // Test that the Sleep/SleepUntil calls do not return until monotonic time
  // passes the requested wakeup time.
  void TestRandomSleep(MonotonicClock* mono_clock) {
    RandomEngine random(testing::UnitTest::GetInstance()->random_seed());
    const int kNumSamples = 5;

    // Sleep.
    for (int i = 0; i < kNumSamples; i++) {
      absl::Duration sleep_time = absl::Seconds(
          std::uniform_real_distribution<float>(0.0f, 0.2f)(random));
      absl::Time before = mono_clock->TimeNow();
      absl::Time wakeup_time = before + sleep_time;
      mono_clock->Sleep(sleep_time);
      absl::Time after = mono_clock->TimeNow();
      ASSERT_LE(wakeup_time, after);
    }

    // SleepUntil.
    for (int i = 0; i < kNumSamples; i++) {
      absl::Duration sleep_time = absl::Seconds(
          std::uniform_real_distribution<float>(0.0f, 0.2f)(random));
      absl::Time before = mono_clock->TimeNow();
      absl::Time wakeup_time = before + sleep_time;
      mono_clock->SleepUntil(wakeup_time);
      absl::Time after = mono_clock->TimeNow();
      ASSERT_LE(wakeup_time, after);
    }
  }

  static State* CreateMonotonicClockState(Clock* raw_clock) {
    return MonotonicClockAccess::CreateMonotonicClockState(raw_clock);
  }

  static MonotonicClock* CreateMonotonicClock(State* state) {
    return MonotonicClockAccess::CreateMonotonicClock(state);
  }

  static void DeleteMonotonicClockState(State* state) {
    MonotonicClockAccess::DeleteMonotonicClockState(state);
  }
};

// Time moves forward only -- there should be no time corrections.
TEST_F(MonotonicClockTest, SimulatedForwardTime) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  TestSimulatedForwardTime(&sim_clock, mono_clock);
  sim_clock.ThreadFinish();
  delete mono_clock;
}

// Time moves forward and backward.
TEST_F(MonotonicClockTest, SimulatedBackwardTime) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  TestSimulatedBackwardTime(&sim_clock, mono_clock);
  sim_clock.ThreadFinish();
  delete mono_clock;
}

// Time moves forward and backward.
TEST_F(MonotonicClockTest, SimulatedTime) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  TestSimulatedBackwardTime(&sim_clock, mono_clock);
  absl::Time mono_time = mono_clock->TimeNow();
  sim_clock.Sleep(absl::Seconds(-1));
  ASSERT_EQ(mono_time, mono_clock->TimeNow());
  sim_clock.ThreadFinish();
  delete mono_clock;
}

// Take a random walk through time.
TEST_F(MonotonicClockTest, SimulatedRandomWalk) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  sim_clock.Sleep(absl::Now() - sim_clock.TimeNow());
  ASSERT_EQ(sim_clock.TimeNow(), mono_clock->TimeNow());

  // Generate kNumSamples random clock adjustments.
  const int kNumSamples = 5;
  RandomEngine random(testing::UnitTest::GetInstance()->random_seed());
  // Keep track of maximum time on clock and corrections.
  absl::Time max_time = sim_clock.TimeNow();
  int num_corrections = 0;
  absl::Duration max_correction = absl::ZeroDuration();
  for (int i = 0; i < kNumSamples; i++) {
    absl::Duration jump =
        absl::Seconds(std::uniform_real_distribution<float>(-0.5, 0.5)(random));
    sim_clock.Sleep(jump);
    absl::Time sim_time = sim_clock.TimeNow();
    if (jump < absl::ZeroDuration()) {
      ASSERT_LT(sim_time, max_time);
      absl::Duration correction = max_time - sim_time;
      if (correction > max_correction) {
        max_correction = correction;
      }
      ++num_corrections;
    }
    if (sim_clock.TimeNow() > max_time) {
      max_time = sim_clock.TimeNow();
    }
    ASSERT_EQ(max_time, mono_clock->TimeNow());
  }
  VerifyCorrectionMetrics(mono_clock, num_corrections,
                          absl::FDivDuration(max_correction, absl::Seconds(1)));
  sim_clock.ThreadFinish();
  delete mono_clock;
}

TEST_F(MonotonicClockTest, RealTime) {
  MonotonicClock* mono_clock =
      MonotonicClock::CreateMonotonicClock(Clock::RealClock());
  // Call mono_clock->Now() continuously for FLAGS_real_test_secs seconds.
  absl::Time start = absl::Now();
  absl::Time time = start;
  int64 num_calls = 0;
  do {
    absl::Time last_time = time;
    time = mono_clock->TimeNow();
    ASSERT_LE(last_time, time);
    ++num_calls;
  } while (time - start < kDefaultRealTest);
  // Just out of curiousity -- did real clock go backwards?
  int clock_num_corrections;
  mono_clock->GetCorrectionMetrics(&clock_num_corrections, NULL);
  LOG(INFO) << clock_num_corrections << " corrections in " << num_calls
            << " calls to mono_clock->Now()";
  delete mono_clock;
}

// Test the Sleep interface using a MonotonicClock.
TEST_F(MonotonicClockTest, RandomSleep) {
  MonotonicClock* mono_clock =
      MonotonicClock::CreateMonotonicClock(Clock::RealClock());
  TestRandomSleep(mono_clock);
  delete mono_clock;
}

// Test the Sleep interface using a SynchronizedMonotonicClock.
TEST_F(MonotonicClockTest, RandomSleepSynced) {
  MonotonicClock* mono_clock =
      MonotonicClock::CreateSynchronizedMonotonicClock();
  TestRandomSleep(mono_clock);
  delete mono_clock;
}

// Test that SleepUntil has no effect if monotonic time has passed the
// requested wakeup time.
TEST_F(MonotonicClockTest, SimulatedInsomnia) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  sim_clock.Sleep(absl::Now() - sim_clock.TimeNow());
  ASSERT_EQ(sim_clock.TimeNow(), mono_clock->TimeNow());

  sim_clock.Sleep(absl::Seconds(-3.14159));
  // Even though sim_clock will never advance, this call will not sleep
  // because monotonic_time has already advanced beyond the wakeup time.
  mono_clock->SleepUntil(sim_clock.TimeNow() + absl::Seconds(1));
  // Note that the same test can't be performed with Sleep because the argument
  // to sleep is an offset from monotonic time, not raw time.
  sim_clock.ThreadFinish();
  delete mono_clock;
}

// Two monotonic clocks, clock1 and clock2, each synced to the same
// raw clock.  Advance simulated time, read one clock, regress simulated
// time, and read the other clock.  The values should be the same.
TEST_F(MonotonicClockTest, SyncedPair) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  State* state = CreateMonotonicClockState(&sim_clock);
  MonotonicClock* clock1 = CreateMonotonicClock(state);
  MonotonicClock* clock2 = CreateMonotonicClock(state);
  sim_clock.Sleep(absl::Seconds(1000));
  ASSERT_EQ(sim_clock.TimeNow(), clock1->TimeNow());
  ASSERT_EQ(sim_clock.TimeNow(), clock2->TimeNow());

  absl::Time time1, time2;
  sim_clock.Sleep(absl::Seconds(2));
  time1 = clock1->TimeNow();
  ASSERT_EQ(sim_clock.TimeNow(), time1);
  sim_clock.Sleep(absl::Seconds(-5));
  time2 = clock2->TimeNow();
  ASSERT_EQ(time1, time2);
  VerifyCorrectionMetrics(clock1, 0, 0.0);
  VerifyCorrectionMetrics(clock2, 1, 5.0);

  clock1->ResetCorrectionMetrics();
  clock2->ResetCorrectionMetrics();
  VerifyCorrectionMetrics(clock1, 0, 0.0);
  VerifyCorrectionMetrics(clock2, 0, 0.0);

  // In this example, time on clock1 goes forward by a greater amount than
  // time goes backward on clock2.  Although clock2 still reports the global
  // monotonic time, it does not report a correction because it never
  // observed a raw clock reading that went backward.
  sim_clock.Sleep(absl::Seconds(10));
  time1 = clock1->TimeNow();
  ASSERT_EQ(sim_clock.TimeNow(), time1);
  sim_clock.Sleep(absl::Seconds(-1));
  time2 = clock2->TimeNow();
  ASSERT_EQ(time1, time2);
  VerifyCorrectionMetrics(clock1, 0, 0.0);
  VerifyCorrectionMetrics(clock2, 0, 0.0);

  sim_clock.ThreadFinish();
  delete clock1;
  delete clock2;
  DeleteMonotonicClockState(state);
}

// Test that a globally-synchronized MonotonicClock is unaffected by clock
// behavior of a vanilla MonotonicClock.
TEST_F(MonotonicClockTest, UnsyncedPair) {
  SimulationClock sim_clock;
  sim_clock.ThreadStart();
  MonotonicClock* sync_clock =
      MonotonicClock::CreateSynchronizedMonotonicClock();
  MonotonicClock* mono_clock = MonotonicClock::CreateMonotonicClock(&sim_clock);
  absl::Time before = sync_clock->TimeNow();
  sim_clock.Sleep(before - sim_clock.TimeNow());
  ASSERT_EQ(before, mono_clock->TimeNow());
  sim_clock.Sleep(absl::Seconds(61));
  ASSERT_LT(sync_clock->TimeNow(), mono_clock->TimeNow());
  sim_clock.ThreadFinish();
  delete sync_clock;
  delete mono_clock;
}

// The factory method CreateSynchronizedMonotonicClock should return a
// MonotonicClock based on real time.  Since time waits for no unit test,
// we can't test equality of the time read from the factory-produced clock
// and the time read from a real clock.  But we can verifying that, as long
// as the real clock moves forward, the time read from the factory-produced
// clock is bounded by consecutive readings of the real clock.
TEST_F(MonotonicClockTest, CreateSynchronizedMonotonicClock) {
  Clock* real_clock = Clock::RealClock();
  MonotonicClock* mono_clock =
      MonotonicClock::CreateSynchronizedMonotonicClock();
  const int kNumSamples = 100;
  for (int i = 0; i < kNumSamples; ++i) {
    absl::Time before = real_clock->TimeNow();
    absl::Time now = mono_clock->TimeNow();
    absl::Time after = real_clock->TimeNow();
    if (after < before) {
      // Real clock moved backward -- test is invalid.
      continue;
    }
    ASSERT_LE(before, now);
    ASSERT_LE(now, after);
  }
  delete mono_clock;
}

// Start up a number of threads to beat on the interface to verify that
// (a) nothing crashes and (b) nothing deadlocks.
class ClockFrenzy {
 public:
  ClockFrenzy()
      : real_clock_(Clock::RealClock()),
        random_(
            new RandomEngine(testing::UnitTest::GetInstance()->random_seed())) {
  }

  void AddSimulationClock(SimulationClock* clock) {
    sim_clocks_.push_back(clock);
  }

  void AddMonotonicClock(MonotonicClock* clock) {
    mono_clocks_.push_back(clock);
  }

  void Feed() {
    while (Running()) {
      // 40% of the time, advance a simulated clock.
      // 50% of the time, read a monotonic clock.
      const int32 u = UniformRandom(100);
      if (u < 40) {
        // Pick a simulated clock and advance it.
        const int nclocks = sim_clocks_.size();
        if (nclocks == 0) continue;
        SimulationClock* sim_clock = sim_clocks_[UniformRandom(nclocks)];
        // Bias the clock towards forward movement.
        sim_clock->Sleep(absl::Seconds(RndFloatRandom() - 0.2));
      } else if (u < 90) {
        // Pick a monotonic clock and read it.
        const int nclocks = mono_clocks_.size();
        if (nclocks == 0) continue;
        MonotonicClock* mono_clock = mono_clocks_[UniformRandom(nclocks)];
        mono_clock->TimeNow();
      }
    }
  }

  // Start Feed-ing threads.
  void Start(int nthreads) {
    absl::MutexLock l(&lock_);
    running_ = true;
    threads_ = absl::make_unique<mediapipe::ThreadPool>("Frenzy", nthreads);
    threads_->StartWorkers();
    for (int i = 0; i < nthreads; ++i) {
      threads_->Schedule([&]() { Feed(); });
    }
  }

  void Stop() {
    absl::MutexLock l(&lock_);
    running_ = false;
  }

  bool Running() {
    absl::MutexLock l(&lock_);
    return running_;
  }

  // Wait for all threads to finish.
  void Wait() { threads_.reset(); }

 private:
  Clock* real_clock_;
  std::vector<SimulationClock*> sim_clocks_;
  std::vector<MonotonicClock*> mono_clocks_;
  std::unique_ptr<mediapipe::ThreadPool> threads_;

  // Provide a lock to avoid race conditions in non-threadsafe ACMRandom.
  mutable absl::Mutex lock_;
  std::unique_ptr<RandomEngine> random_ ABSL_GUARDED_BY(lock_);

  // The stopping notification.
  bool running_;

  // Thread-safe random number generation functions for use by other class
  // member functions.
  int32 UniformRandom(int32 n) {
    absl::MutexLock l(&lock_);
    return std::uniform_int_distribution<int32>(0, n - 1)(*random_);
  }

  float RndFloatRandom() {
    absl::MutexLock l(&lock_);
    return std::uniform_real_distribution<float>(0.0f, 1.0f)(*random_);
  }
};

TEST_F(MonotonicClockTest, SimulatedFrenzy) {
  ClockFrenzy f;
  SimulationClock s1, s2;
  s1.ThreadStart();
  s2.ThreadStart();
  f.AddSimulationClock(&s1);
  f.AddSimulationClock(&s2);
  MonotonicClock* m11 = MonotonicClock::CreateMonotonicClock(&s1);
  State* state = CreateMonotonicClockState(&s1);
  MonotonicClock* m12 = CreateMonotonicClock(state);
  MonotonicClock* m13 = CreateMonotonicClock(state);
  MonotonicClock* m21 = MonotonicClock::CreateMonotonicClock(&s2);
  MonotonicClock* m22 = MonotonicClock::CreateMonotonicClock(&s2);
  f.AddMonotonicClock(m11);
  f.AddMonotonicClock(m12);
  f.AddMonotonicClock(m13);
  f.AddMonotonicClock(m21);
  f.AddMonotonicClock(m22);
  f.Start(10);
  Clock::RealClock()->Sleep(absl::Seconds(1));
  f.Stop();
  f.Wait();
  s2.ThreadFinish();
  s1.ThreadFinish();
  delete m11;
  delete m12;
  delete m13;
  delete m21;
  delete m22;
  DeleteMonotonicClockState(state);
}

// Just for completeness, a frenzy with only real-time
// SynchronizedMonotonicClock instances.
TEST_F(MonotonicClockTest, RealFrenzy) {
  ClockFrenzy f;
  MonotonicClock* m1 = MonotonicClock::CreateSynchronizedMonotonicClock();
  MonotonicClock* m2 = MonotonicClock::CreateSynchronizedMonotonicClock();
  MonotonicClock* m3 = MonotonicClock::CreateSynchronizedMonotonicClock();
  f.AddMonotonicClock(m1);
  f.AddMonotonicClock(m2);
  f.AddMonotonicClock(m3);
  f.Start(10);
  Clock::RealClock()->Sleep(kDefaultRealTest);
  f.Stop();
  f.Wait();
  // Just out of curiousity -- did real clock go backwards?
  int clock_num_corrections;
  m1->GetCorrectionMetrics(&clock_num_corrections, NULL);
  LOG_IF(INFO, clock_num_corrections > 0)
      << clock_num_corrections << " corrections";
  m2->GetCorrectionMetrics(&clock_num_corrections, NULL);
  LOG_IF(INFO, clock_num_corrections > 0)
      << clock_num_corrections << " corrections";
  m3->GetCorrectionMetrics(&clock_num_corrections, NULL);
  LOG_IF(INFO, clock_num_corrections > 0)
      << clock_num_corrections << " corrections";
  delete m1;
  delete m2;
  delete m3;
}

}  // namespace mediapipe
