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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * MediaPipe audio task running mode. A MediaPipe audio task can be run with three different
 * modes: image, video and live stream.
 */
typedef NS_ENUM(NSUInteger, MPPAudioRunningMode) {

  /** The mode for running a mediapipe audio task on independent audio clips. */
  MPPAudioRunningModeAudioClips NS_SWIFT_NAME(audioClips),

      /**
       * The mode for running a mediapipe audio task on an audio stream, such as from a microphone.
       */
      MPPAudioRunningModeAudioStream NS_SWIFT_NAME(audioStream),

  } NS_SWIFT_NAME(RunningMode);  // In Swift `RunningMode` can be resolved as
                                 // `MediaPipeTasksAudio.RunningMode` when used alongside the other
                                 // task libraries.

NS_INLINE NSString *MPPAudioRunningModeDisplayName(MPPAudioRunningMode runningMode) {
  switch (runningMode) {
    case MPPAudioRunningModeAudioClips:
      return @"Audio Clips";
    case MPPAudioRunningModeAudioStream:
      return @"Audio Stream";
    default:
      return @"";
  }
}

NS_ASSUME_NONNULL_END
