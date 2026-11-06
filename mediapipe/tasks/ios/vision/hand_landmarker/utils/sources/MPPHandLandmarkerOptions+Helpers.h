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

#import "mediapipe/tasks/ios/vision/hand_landmarker/sources/MPPHandLandmarkerOptions.h"

struct MpHandLandmarkerOptions;

NS_ASSUME_NONNULL_BEGIN

@interface MPPHandLandmarkerOptions (Helpers)

/** Copies the options to a MediaPipe C API `MpHandLandmarkerOptions` struct. */
- (void)copyToCOptions:(struct MpHandLandmarkerOptions *)cOptions;

/** Frees any memory allocated inside a MediaPipe C API `MpHandLandmarkerOptions` struct. */
+ (void)freeCOptions:(struct MpHandLandmarkerOptions *)cOptions;

@end

NS_ASSUME_NONNULL_END
