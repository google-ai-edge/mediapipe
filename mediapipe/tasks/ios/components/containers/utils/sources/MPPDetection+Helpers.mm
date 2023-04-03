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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPDetection+Helpers.h"
#import "mediapipe/framework/formats/location_data.pb.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"

static const NSInteger kDefaultCategoryIndex = -1;

namespace {
using DetectionProto = ::mediapipe::Detection;
using BoundingBoxProto = ::mediapipe::LocationData::BoundingBox;
}  // namespace

@implementation MPPDetection (Helpers)

+ (MPPDetection *)detectionWithProto:(const DetectionProto &)detectionProto {
  NSMutableArray<MPPCategory *> *categories =
      [NSMutableArray arrayWithCapacity:(NSUInteger)detectionProto.score_size()];

  for (int idx = 0; idx < detectionProto.score_size(); ++idx) {
    NSInteger categoryIndex =
        detectionProto.label_id_size() > idx ? detectionProto.label_id(idx) : kDefaultCategoryIndex;
    NSString *categoryName = detectionProto.label_size() > idx
                                 ? [NSString stringWithCppString:detectionProto.label(idx)]
                                 : nil;

    NSString *displayName = detectionProto.display_name_size() > idx
                                ? [NSString stringWithCppString:detectionProto.display_name(idx)]
                                : nil;

    [categories addObject:[[MPPCategory alloc] initWithIndex:categoryIndex
                                                       score:detectionProto.score(idx)
                                                categoryName:categoryName
                                                 displayName:displayName]];
  }

  CGRect boundingBox = CGRectZero;

  if (detectionProto.location_data().has_bounding_box()) {
    const BoundingBoxProto &boundingBoxProto = detectionProto.location_data().bounding_box();
    boundingBox.origin.x = boundingBoxProto.xmin();
    boundingBox.origin.y = boundingBoxProto.ymin();
    boundingBox.size.width = boundingBoxProto.width();
    boundingBox.size.height = boundingBoxProto.height();
  }

  NSMutableArray<MPPNormalizedKeypoint *> *normalizedKeypoints;

  if (!detectionProto.location_data().relative_keypoints().empty()) {
    normalizedKeypoints = [NSMutableArray
        arrayWithCapacity:(NSUInteger)detectionProto.location_data().relative_keypoints_size()];
    for (const auto &keypoint : detectionProto.location_data().relative_keypoints()) {
      NSString *label = keypoint.has_keypoint_label()
                            ? [NSString stringWithCppString:keypoint.keypoint_label()]
                            : nil;
      CGPoint location = CGPointMake(keypoint.x(), keypoint.y());
      float score = keypoint.has_score() ? keypoint.score() : 0.0f;

      [normalizedKeypoints addObject:[[MPPNormalizedKeypoint alloc] initWithLocation:location
                                                                               label:label
                                                                               score:score]];
    }
  }

  return [[MPPDetection alloc] initWithCategories:categories
                                      boundingBox:boundingBox
                                        keypoints:normalizedKeypoints];
}

@end
