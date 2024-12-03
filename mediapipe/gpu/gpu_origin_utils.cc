#include "mediapipe/gpu/gpu_origin_utils.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace mediapipe {

absl::StatusOr<bool> IsGpuOriginAtBottom(mediapipe::GpuOrigin::Mode origin) {
  switch (origin) {
    case mediapipe::GpuOrigin::TOP_LEFT:
      return false;
    case mediapipe::GpuOrigin::DEFAULT:
    case mediapipe::GpuOrigin::CONVENTIONAL:
      // TOP_LEFT on Metal, BOTTOM_LEFT on OpenGL.
#ifdef __APPLE__
      return false;
#else
      return true;
#endif
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unhandled GPU origin %i", origin));
  }
}

}  // namespace mediapipe
