// Copyright 2023 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef MEDIAPIPE_OBJC_AUDIO_UTIL_H_
#define MEDIAPIPE_OBJC_AUDIO_UTIL_H_

#import <CoreAudio/CoreAudioTypes.h>
#import <CoreMedia/CoreMedia.h>

#include <memory>

#include "absl/status/statusor.h"
#include "mediapipe/framework/formats/matrix.h"

NS_ASSUME_NONNULL_BEGIN

// Converts an audio sample buffer list into a `mediapipe::Matrix`.
// Returns an error status on failure.
absl::StatusOr<std::unique_ptr<mediapipe::Matrix>> MediaPipeConvertAudioBufferListToAudioMatrix(
    const AudioBufferList* audioBufferList, const AudioStreamBasicDescription* streamHeader,
    CMItemCount numFrames);

NS_ASSUME_NONNULL_END

#endif  // MEDIAPIPE_OBJC_AUDIO_UTIL_H_
