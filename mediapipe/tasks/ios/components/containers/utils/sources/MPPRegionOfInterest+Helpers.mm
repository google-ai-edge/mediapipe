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

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPRegionOfInterest+Helpers.h"

#include "mediapipe/tasks/cc/vision/interactive_segmenter/proto/region_of_interest.pb.h"
#include "mediapipe/util/color.pb.h"

namespace {
using RenderData = ::mediapipe::RenderData;
using RenderAnnotation = ::mediapipe::RenderAnnotation;
using RegionOfInterest = ::mediapipe::tasks::vision::interactive_segmenter::proto::RegionOfInterest;
}  // namespace

@implementation MPPRegionOfInterest (Helpers)

- (std::optional<RegionOfInterest>)getRegionOfInteresProtoWithError:(NSError**)error {
  RegionOfInterest result;
  if (self.keypoint) {
    auto* point = result.mutable_keypoint();
    point->set_normalized(true);
    point->set_x(self.keypoint.location.x);
    point->set_y(self.keypoint.location.y);
    return result;
  } else if (self.scribbles) {
    auto* scribble = result.mutable_scribble();
    for (MPPNormalizedKeypoint* keypoint in self.scribbles) {
      auto* point = scribble->add_point();
      point->set_normalized(true);
      point->set_x(keypoint.location.x);
      point->set_y(keypoint.location.y);
    }
    return result;
  }

  [MPPCommonUtils createCustomError:error
                           withCode:MPPTasksErrorCodeInvalidArgumentError
                        description:@"RegionOfInterest does not include a valid user interaction."];

  return std::nullopt;
}

@end
