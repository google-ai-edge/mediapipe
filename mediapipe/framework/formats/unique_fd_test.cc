#include "mediapipe/framework/formats/unique_fd.h"

#include <fcntl.h>
#include <unistd.h>

#include <utility>

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

namespace {

// Returns a valid system file descriptor.
int GetValidFd() { return dup(STDOUT_FILENO); }

// Helper function to check if the file descriptor is valid (still open).
int IsFdValid(int fd) { return fcntl(fd, F_GETFD) != -1; }

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

TEST(UniqueFdTest, ShouldReleaseValidFd) {
  UniqueFd unique_fd(GetValidFd());
  EXPECT_TRUE(unique_fd.IsValid());

  const int released_fd = unique_fd.Release();

  EXPECT_FALSE(unique_fd.IsValid());
  close(released_fd);
}

}  // namespace

}  // namespace mediapipe
