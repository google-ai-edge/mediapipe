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
 * MediaPipe vision task running mode. A MediaPipe vision task can be run with three different
 * modes: image, video and live stream.
 */
typedef NS_ENUM(NSUInteger, MPPRunningMode) {

  // Generic error codes.

  /** The mode for running a mediapipe vision task on single image inputs. */
  MPPRunningModeImage,

  /** The mode for running a mediapipe vision task on the decoded frames of a video. */
  MPPRunningModeVideo,

  /**
   * The mode for running a mediapipe vision task on a live stream of input data, such as from the
   * camera.
   */
  MPPRunningModeLiveStream,

} NS_SWIFT_NAME(RunningMode);

NS_INLINE NSString *MPPRunningModeDisplayName(MPPRunningMode runningMode) {
  switch (runningMode) {
    case MPPRunningModeImage:
      return @"Image";
    case MPPRunningModeVideo:
      return @"Video";
    case MPPRunningModeLiveStream:
      return @"Live Stream";
    default:
      return nil;
  }
}

NS_ASSUME_NONNULL_END
