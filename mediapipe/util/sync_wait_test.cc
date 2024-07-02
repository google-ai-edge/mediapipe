#include "mediapipe/util/sync_wait.h"

#include <fcntl.h>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "mediapipe/framework/port.h"  // IWYU pragma: keep (DRIHSTI_OSX)
#include "mediapipe/framework/port/benchmark.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"

#ifdef MEDIAPIPE_OSX
#include <sys/event.h>
#else
#include <sys/timerfd.h>
#endif  // MEDIAPIPE_OSX

namespace mediapipe {
namespace {

struct TestTimer {
  TestTimer() = default;
  ~TestTimer() {
    if (fd != -1) {
      ABSL_CHECK_EQ(close(fd), 0);
    }
  }
  TestTimer(TestTimer&& timer) = default;
  TestTimer& operator=(TestTimer&& timer) = default;

  int fd = -1;
};

#ifdef MEDIAPIPE_OSX
TestTimer CreateTestTimer(absl::Duration duration) {
  int kq = kqueue();
  ABSL_CHECK_NE(kq, -1);

  struct kevent kev;
  constexpr int kTimerId = 1;
  const int timeout = static_cast<int>(absl::ToInt64Milliseconds(duration));
  EV_SET(&kev, kTimerId, EVFILT_TIMER, EV_ADD | EV_ENABLE | EV_ONESHOT,
         NOTE_CRITICAL, timeout, NULL);
  kevent(kq, &kev, 1, NULL, 0, NULL);

  TestTimer timer;
  timer.fd = kq;
  return timer;
}
#else
TestTimer CreateTestTimer(absl::Duration duration) {
  TestTimer timer;
  timer.fd = timerfd_create(CLOCK_MONOTONIC, /*flags*/ 0);
  ABSL_CHECK_NE(timer.fd, -1);

  struct itimerspec new_value;
  new_value.it_value = absl::ToTimespec(duration);
  new_value.it_interval.tv_sec = 0;
  new_value.it_interval.tv_nsec = 0;

  ABSL_CHECK_NE(
      timerfd_settime(timer.fd, /*flags=*/0, &new_value, /*oldtimer=*/nullptr),
      -1);

  return timer;
}
#endif  // MEDIAPIPE_OSX

TEST(SyncWait, WorksWithIndefiniteTimeout) {
  TestTimer timer = CreateTestTimer(absl::Milliseconds(2));
  MP_EXPECT_OK(mediapipe::SyncWait(timer.fd, absl::InfiniteDuration()));
}

TEST(SyncWait, WorksWithDefiniteTimeout) {
  TestTimer timer = CreateTestTimer(absl::Milliseconds(5));
  MP_EXPECT_OK(mediapipe::SyncWait(timer.fd, absl::Milliseconds(10)));
}

TEST(SyncWait, WorksWithReadyFd) {
  TestTimer timer = CreateTestTimer(absl::Milliseconds(5));
  // timer.fd is not available for read
  MP_EXPECT_OK(mediapipe::SyncWait(timer.fd, absl::InfiniteDuration()));

  // timer.fd is available for read
  MP_EXPECT_OK(mediapipe::SyncWait(timer.fd, absl::InfiniteDuration()));
  MP_EXPECT_OK(mediapipe::SyncWait(timer.fd, absl::Milliseconds(1)));
}

TEST(SyncWait, ReportsTimeout) {
  TestTimer timer = CreateTestTimer(absl::Milliseconds(100));
  EXPECT_THAT(mediapipe::SyncWait(timer.fd, absl::Milliseconds(5)),
              StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(SyncWait, ReportsInvalidFd) {
  const int fd = -1;
  EXPECT_THAT(mediapipe::SyncWait(fd, absl::InfiniteDuration()),
              StatusIs(absl::StatusCode::kInternal));
}

void BM_SyncWaitZeroTimeout(benchmark::State& state) {
  // Non blocking waits will be used and timer canceled automatically after
  // benchmark completion.
  TestTimer timer = CreateTestTimer(absl::Minutes(1));
  for (auto s : state) {
    ABSL_CHECK_EQ(mediapipe::SyncWait(timer.fd, absl::ZeroDuration()).code(),
                  absl::StatusCode::kDeadlineExceeded);
  }
}
BENCHMARK(BM_SyncWaitZeroTimeout);

}  // namespace
}  // namespace mediapipe
