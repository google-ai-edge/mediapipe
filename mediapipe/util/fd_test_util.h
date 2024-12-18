#ifndef MEDIAPIPE_UTIL_FD_TEST_UTIL_H_
#define MEDIAPIPE_UTIL_FD_TEST_UTIL_H_

#include <fcntl.h>
#include <unistd.h>

namespace mediapipe {

// Returns a valid system file descriptor.
inline int GetValidFd() { return dup(STDOUT_FILENO); }

// Helper function to check if the file descriptor is valid (still open).
inline int IsFdValid(int fd) { return fcntl(fd, F_GETFD) != -1; }

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_FD_TEST_UTIL_H_
