// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierResult.h"

struct MpAudioClassifierResult;

NS_ASSUME_NONNULL_BEGIN

@interface MPPAudioClassifierResult (Helpers)

/**
 * Creates an `MPPAudioClassifierResult` from a C `MpAudioClassifierResult` struct and timestamp.
 *
 * @param cResult The C `MpAudioClassifierResult` struct.
 * @param timestampInMilliseconds The timestamp in milliseconds.
 *
 * @return An `MPPAudioClassifierResult` object containing classification results.
 */
+ (nullable MPPAudioClassifierResult *)
    audioClassifierResultWithCResult:(const struct MpAudioClassifierResult &)cResult
             timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
