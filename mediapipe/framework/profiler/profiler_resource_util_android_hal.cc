#include "absl/status/status.h"    // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace mediapipe {

absl::StatusOr<std::string> GetDefaultTraceLogDirectory() {
  return "/data/local/tmp";
}

}  // namespace mediapipe
