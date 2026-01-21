#ifndef MEDIAPIPE_TASKS_C_AUDIO_CORE_RUNNING_MODE_CONVERTER_H_
#define MEDIAPIPE_TASKS_C_AUDIO_CORE_RUNNING_MODE_CONVERTER_H_

#include "absl/status/statusor.h"
#include "mediapipe/tasks/c/audio/core/common.h"
#include "mediapipe/tasks/cc/audio/core/running_mode.h"

namespace mediapipe::tasks::c::audio::core {

// Converts a C RunningMode to its C++ equivalent.
absl::StatusOr<mediapipe::tasks::audio::core::RunningMode>
CppConvertToRunningMode(MpAudioRunningMode running_mode);

}  // namespace mediapipe::tasks::c::audio::core

#endif  // MEDIAPIPE_TASKS_C_AUDIO_CORE_RUNNING_MODE_CONVERTER_H_
