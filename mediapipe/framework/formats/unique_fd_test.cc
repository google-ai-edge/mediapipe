#include "mediapipe/framework/formats/unique_fd.h"

#include <utility>

#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/status_matchers.h"
#include "mediapipe/util/fd_test_util.h"

namespace mediapipe {

namespace {

TEST(UniqueFdTest, ShouldInitializeInvalidFd) {
  UniqueFd unique_fd;
  EXPECT_FALSE(unique_fd.IsValid());
}

TEST(UniqueFdTest, ShouldWrapFd) {
  const int fd = GetValidFd();

  UniqueFd unique_fd(fd);

  EXPECT_EQ(unique_fd.Get(), fd);
}

TEST(UniqueFdTest, ShouldCloseFdDuringDestruction) {
  const int fd = GetValidFd();
  EXPECT_TRUE(IsFdValid(fd));

  {
    UniqueFd unique_fd(fd);
  }

  EXPECT_FALSE(IsFdValid(fd));
}

TEST(UniqueFdTest, ShouldMoveUniqueFd) {
  const int fd = GetValidFd();
  UniqueFd unique_fd(fd);
  EXPECT_TRUE(unique_fd.IsValid());

  UniqueFd moved_unique_fd = std::move(unique_fd);

  EXPECT_TRUE(moved_unique_fd.IsValid());
  EXPECT_EQ(moved_unique_fd.Get(), fd);
}

TEST(UniqueFdTest, ShouldCreateValidFd) {
  UniqueFd unique_fd(GetValidFd());
  EXPECT_TRUE(unique_fd.IsValid());

  unique_fd.Reset();
  EXPECT_FALSE(unique_fd.IsValid());
}

TEST(UniqueFdTest, ShouldDupValidFd) {
  UniqueFd unique_fd(GetValidFd());

  MP_ASSERT_OK_AND_ASSIGN(UniqueFd dup_unique_fd, unique_fd.Dup());

  EXPECT_TRUE(dup_unique_fd.IsValid());
  EXPECT_NE(dup_unique_fd.Get(), unique_fd.Get());
}

TEST(UniqueFdTest, ShouldReleaseValidFd) {
  UniqueFd unique_fd(GetValidFd());
  EXPECT_TRUE(unique_fd.IsValid());

  const int released_fd = unique_fd.Release();

  EXPECT_FALSE(unique_fd.IsValid());
  close(released_fd);
}

}  // namespace

}  // namespace mediapipe
