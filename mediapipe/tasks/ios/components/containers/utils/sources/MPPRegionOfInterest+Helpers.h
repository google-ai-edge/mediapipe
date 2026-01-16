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

#include "mediapipe/util/render_data.pb.h"

#import "mediapipe/tasks/ios/components/containers/sources/MPPRegionOfInterest.h"

#include "mediapipe/tasks/cc/vision/interactive_segmenter/proto/region_of_interest.pb.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPRegionOfInterest (Helpers)

/**
 * Creates a `RegionOfInterest` proto from the region of interest.
 *
 * @param error Pointer to the memory location where errors if any should be saved. If @c NULL, no
 * error will be saved.
 *
 * @return A `RegionOfInterest` proto created from the region of interest.
 */
- (std::optional<mediapipe::tasks::vision::interactive_segmenter::proto::RegionOfInterest>)
    getRegionOfInteresProtoWithError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
