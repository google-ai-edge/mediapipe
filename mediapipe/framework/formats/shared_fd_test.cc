#include "mediapipe/framework/formats/shared_fd.h"

#include <utility>

#include "mediapipe/framework/formats/unique_fd.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/fd_test_util.h"

namespace mediapipe {
namespace {

TEST(SharedFdTest, CanCreateFromUniqueFd) {
  int raw_fd = GetValidFd();
  {
    auto fd = SharedFd(UniqueFd(raw_fd));
    EXPECT_TRUE(IsFdValid(fd.Get()));
  }
  EXPECT_FALSE(IsFdValid(raw_fd));
}

TEST(SharedFdTest, CanCopyAndMoveFd) {
  int raw_fd = GetValidFd();
  auto fd = SharedFd(UniqueFd(raw_fd));
  {
    SharedFd copied_fd = fd;
    EXPECT_TRUE(IsFdValid(copied_fd.Get()));
  }
  EXPECT_TRUE(IsFdValid(fd.Get()));

  {
    SharedFd moved_fd = std::move(fd);
    EXPECT_TRUE(IsFdValid(moved_fd.Get()));
  }
  EXPECT_FALSE(IsFdValid(raw_fd));
}

TEST(SharedFdTest, CanBeAssignedAndComparedWithNullptr) {
  SharedFd fd;
  EXPECT_FALSE(fd);
  EXPECT_EQ(fd, nullptr);

  int raw_fd = GetValidFd();
  fd = SharedFd(UniqueFd(raw_fd));

  EXPECT_NE(fd, nullptr);
  EXPECT_TRUE(fd);

  fd = nullptr;
  EXPECT_FALSE(IsFdValid(raw_fd));
  EXPECT_EQ(fd, nullptr);
  EXPECT_FALSE(fd);
}

TEST(SharedFdTest, CanDup) {
  int raw_fd = GetValidFd();
  auto fd = SharedFd(UniqueFd(GetValidFd()));
  MP_ASSERT_OK_AND_ASSIGN(UniqueFd dup_fd, fd.Dup());
  EXPECT_NE(dup_fd.Get(), raw_fd);
  EXPECT_TRUE(IsFdValid(dup_fd.Get()));
}

}  // namespace
}  // namespace mediapipe
