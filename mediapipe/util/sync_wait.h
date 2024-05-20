#ifndef MEDIAPIPE_UTIL_SYNC_WAIT_H_
#define MEDIAPIPE_UTIL_SYNC_WAIT_H_

#include "absl/status/status.h"
#include "absl/time/time.h"

namespace mediapipe {

// `fd` - represents a sync
// `timeout` - wait timeout, pass `absl::InfiniteDuration()` to wait until
//             signaled.
absl::Status SyncWait(int fd, absl::Duration timeout);

}  // namespace mediapipe

#endif  // MEDIAPIPE_UTIL_SYNC_WAIT_H_
