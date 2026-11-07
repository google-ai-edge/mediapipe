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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/sources/MPPGestureRecognizerResult.h"

struct MpGestureRecognizerResult;

NS_ASSUME_NONNULL_BEGIN

@interface MPPGestureRecognizerResult (Helpers)

/**
 * Creates an `MPPGestureRecognizerResult` from a MediaPipe C API `MpGestureRecognizerResult`
 * struct.
 *
 * @param cResult The C API gesture recognizer result struct.
 * @param timestampInMilliseconds The timestamp in milliseconds.
 *
 * @return An `MPPGestureRecognizerResult` object.
 */
+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultWithCResult:(const struct MpGestureRecognizerResult &)cResult
               timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
