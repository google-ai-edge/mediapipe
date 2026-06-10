/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MEDIAPIPE_TASKS_CC_AUDIO_CORE_RUNNING_MODE_H_
#define MEDIAPIPE_TASKS_CC_AUDIO_CORE_RUNNING_MODE_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/tasks/cc/core/running_mode.h"

namespace mediapipe {
namespace tasks {
namespace audio {
namespace core {

// The running mode of a MediaPipe audio task.
enum RunningMode {
  // Run the audio task on independent audio clips.
  AUDIO_CLIPS = 1,

  // Run the audio task on an audio stream, such as from microphone.
  AUDIO_STREAM = 2,
};

inline std::string GetRunningModeName(RunningMode mode) {
  switch (mode) {
    case AUDIO_CLIPS:
      return "audio clips mode";
    case AUDIO_STREAM:
      return "audio stream mode";
    default:
      return "unknown mode";
  }
}

inline mediapipe::tasks::core::RunningMode GetCoreRunningMode(
    RunningMode mode) {
  switch (mode) {
    case AUDIO_CLIPS:
      return mediapipe::tasks::core::RunningMode::kAudioClips;
    case AUDIO_STREAM:
      return mediapipe::tasks::core::RunningMode::kAudioStream;
    default:
      return mediapipe::tasks::core::RunningMode::kUnspecified;
  }
}

inline absl::StatusOr<RunningMode> GetAudioRunningMode(
    mediapipe::tasks::core::RunningMode mode_name) {
  if (mode_name == mediapipe::tasks::core::RunningMode::kAudioClips) {
    return AUDIO_CLIPS;
  } else if (mode_name == mediapipe::tasks::core::RunningMode::kAudioStream) {
    return AUDIO_STREAM;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported running mode: ", mode_name));
  }
}

}  // namespace core
}  // namespace audio
}  // namespace tasks
}  // namespace mediapipe

#endif  // MEDIAPIPE_TASKS_CC_AUDIO_CORE_RUNNING_MODE_H_
