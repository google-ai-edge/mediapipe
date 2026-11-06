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

#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerResult.h"

struct MpHandLandmarkerResult;

NS_ASSUME_NONNULL_BEGIN

@interface MPPHandLandmarkerResult (Helpers)

/**
 * Creates an `MPPHandLandmarkerResult` from a MediaPipe C API `MpHandLandmarkerResult` struct.
 *
 * @param cResult The C API hand landmarker result struct.
 * @param timestampInMilliseconds The timestamp in milliseconds.
 *
 * @return An `MPPHandLandmarkerResult` object.
 */
+ (MPPHandLandmarkerResult *)handLandmarkerResultWithCResult:
                                 (const struct MpHandLandmarkerResult &)cResult
                                     timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
