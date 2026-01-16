#include "mediapipe/tasks/c/audio/core/running_mode_converter.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/cc/audio/core/running_mode.h"

namespace mediapipe::tasks::c::audio::core {

absl::StatusOr<mediapipe::tasks::audio::core::RunningMode>
CppConvertToRunningMode(MpAudioRunningMode running_mode) {
  switch (running_mode) {
    case kMpAudioRunningModeAudioClips:
      return mediapipe::tasks::audio::core::RunningMode::AUDIO_CLIPS;
    case kMpAudioRunningModeAudioStream:
      return mediapipe::tasks::audio::core::RunningMode::AUDIO_STREAM;
    default:
      return absl::InvalidArgumentError("Unsupported running mode");
  }
}

}  // namespace mediapipe::tasks::c::audio::core
