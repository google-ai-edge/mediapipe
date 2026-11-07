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

#import "mediapipe/tasks/ios/components/containers/sources/MPPLandmark.h"

struct MpLandmark;
struct MpLandmarks;
struct MpNormalizedLandmark;
struct MpNormalizedLandmarks;

NS_ASSUME_NONNULL_BEGIN

@interface MPPLandmark (Helpers)

+ (MPPLandmark *)landmarkWithCLandmark:(const struct MpLandmark &)cLandmark;

+ (NSArray<MPPLandmark *> *)landmarksWithCLandmarks:(const struct MpLandmarks &)cLandmarks;

@end

@interface MPPNormalizedLandmark (Helpers)

+ (MPPNormalizedLandmark *)normalizedLandmarkWithCNormalizedLandmark:
    (const struct MpNormalizedLandmark &)cNormalizedLandmark;

+ (NSArray<MPPNormalizedLandmark *> *)normalizedLandmarksWithCNormalizedLandmarks:
    (const struct MpNormalizedLandmarks &)cNormalizedLandmarks;

@end

NS_ASSUME_NONNULL_END
