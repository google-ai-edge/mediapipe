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

#import "mediapipe/tasks/ios/test/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+ProtobufHelpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPClassificationResult.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPLandmark.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/ios/test/vision/utils/sources/parse_proto_utils.h"

namespace {
using mediapipe::NormalizedLandmarkList;
using ::mediapipe::tasks::ios::test::vision::utils::get_proto_from_pbtxt;
using ::mediapipe::tasks::vision::holistic_landmarker::proto::HolisticResult;

NSArray<MPPNormalizedLandmark *> *LandmarksArrayFromProto(const NormalizedLandmarkList &protoList) {
  NSMutableArray<MPPNormalizedLandmark *> *landmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)protoList.landmark().size()];
  for (const auto &landmark : protoList.landmark()) {
    NSNumber *visibility = landmark.has_visibility() ? @(landmark.visibility()) : nil;
    NSNumber *presence = landmark.has_presence() ? @(landmark.presence()) : nil;
    [landmarks addObject:[[MPPNormalizedLandmark alloc] initWithX:landmark.x()
                                                                y:landmark.y()
                                                                z:landmark.z()
                                                       visibility:visibility
                                                         presence:presence]];
  }
  return landmarks;
}
}  // anonymous namespace

@implementation MPPHolisticLandmarkerResult (ProtobufHelpers)

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultFromProtobufFileWithName:(NSString *)fileName
                                  hasFaceBlendshapes:(BOOL)hasFaceBlendshapes {
  HolisticResult holisticResultProto;

  if (!get_proto_from_pbtxt(fileName.cppString, holisticResultProto).ok()) {
    return nil;
  }

  NSArray<MPPNormalizedLandmark *> *faceLandmarks =
      LandmarksArrayFromProto(holisticResultProto.face_landmarks());
  NSArray<MPPNormalizedLandmark *> *poseLandmarks =
      LandmarksArrayFromProto(holisticResultProto.pose_landmarks());
  NSArray<MPPNormalizedLandmark *> *leftHandLandmarks =
      LandmarksArrayFromProto(holisticResultProto.left_hand_landmarks());
  NSArray<MPPNormalizedLandmark *> *rightHandLandmarks =
      LandmarksArrayFromProto(holisticResultProto.right_hand_landmarks());

  MPPClassifications *faceBlendshapes = nil;
  if (hasFaceBlendshapes && holisticResultProto.has_face_blendshapes()) {
    NSMutableArray<MPPCategory *> *categories =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(
                                              holisticResultProto.face_blendshapes()
                                                  .classification()
                                                  .size())];
    for (const auto &c : holisticResultProto.face_blendshapes().classification()) {
      NSString *categoryName =
          c.has_label() ? [NSString stringWithUTF8String:c.label().c_str()] : nil;
      NSString *displayName =
          c.has_display_name() ? [NSString stringWithUTF8String:c.display_name().c_str()] : nil;
      [categories addObject:[[MPPCategory alloc] initWithIndex:c.index()
                                                         score:c.score()
                                                  categoryName:categoryName
                                                   displayName:displayName]];
    }
    faceBlendshapes = [[MPPClassifications alloc] initWithHeadIndex:0
                                                           headName:@""
                                                         categories:categories];
  }

  return [[MPPHolisticLandmarkerResult alloc] initWithFaceLandmarks:faceLandmarks
                                                    faceBlendshapes:faceBlendshapes
                                                      poseLandmarks:poseLandmarks
                                                 poseWorldLandmarks:@[]
                                               poseSegmentationMask:nil
                                                  leftHandLandmarks:leftHandLandmarks
                                             leftHandWorldLandmarks:@[]
                                                 rightHandLandmarks:rightHandLandmarks
                                            rightHandWorldLandmarks:@[]
                                            timestampInMilliseconds:0];
}

@end
