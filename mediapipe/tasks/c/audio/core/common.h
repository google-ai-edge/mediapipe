/* Copyright 2025 The MediaPipe Authors.

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

#ifndef MEDIAPIPE_TASKS_C_ADUIO_CORE_COMMON_H_
#define MEDIAPIPE_TASKS_C_ADUIO_CORE_COMMON_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Supported processing modes.
enum MpAudioRunningMode {
  kMpAudioRunningModeAudioClips = 1,
  kMpAudioRunningModeAudioStream = 2,
};

// MediaPipe audio data.
struct MpAudioData {
  // The number of channels.
  int num_channels;
  // The sample rate.
  double sample_rate;
  // The audio data in a row-major matrix, where rows contain the samples with
  // a column for each channel.
  float* audio_data;
  // The size of the audio data buffer.
  size_t audio_data_size;
};

#ifdef __cplusplus
}  // extern C
#endif

#endif  // MEDIAPIPE_TASKS_C_ADUIO_CORE_COMMON_H_
