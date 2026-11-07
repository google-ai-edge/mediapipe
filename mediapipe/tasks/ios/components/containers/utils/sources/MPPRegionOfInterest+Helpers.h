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

#import "mediapipe/tasks/ios/components/containers/sources/MPPRegionOfInterest.h"

struct MpRegionOfInterest;

NS_ASSUME_NONNULL_BEGIN

@interface MPPRegionOfInterest (Helpers)

/**
 * Copies the region of interest to a C MpRegionOfInterest struct.
 * The caller is responsible for freeing allocated scribble memory by calling
 * +[MPPRegionOfInterest freeCRegionOfInterest:].
 */
- (BOOL)copyToCRegionOfInterest:(struct MpRegionOfInterest *)cRoi error:(NSError **)error;

/** Frees any memory allocated inside MpRegionOfInterest. */
+ (void)freeCRegionOfInterest:(nullable struct MpRegionOfInterest *)cRoi;

@end

NS_ASSUME_NONNULL_END
