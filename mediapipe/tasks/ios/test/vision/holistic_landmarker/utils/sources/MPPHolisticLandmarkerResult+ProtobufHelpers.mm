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
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

#include "mediapipe/tasks/cc/vision/holistic_landmarker/proto/holistic_result.pb.h"
#include "mediapipe/tasks/ios/test/vision/utils/sources/parse_proto_utils.h"

namespace {
using mediapipe::ClassificationList;
using mediapipe::LandmarkList;
using ::mediapipe::tasks::ios::test::vision::utils::get_proto_from_pbtxt;
using ::mediapipe::tasks::vision::holistic_landmarker::proto::HolisticResult;
};  // anonymous namespace

@implementation MPPHolisticLandmarkerResult (ProtobufHelpers)

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultFromProtobufFileWithName:(NSString *)fileName
                                  hasFaceBlendshapes:(BOOL)hasFaceBlenshapes {
  HolisticResult holisticResultProto;

  if (!get_proto_from_pbtxt(fileName.cppString, holisticResultProto).ok()) {
    return nil;
  }

  const ClassificationList *faceBlendshapesProto =
      hasFaceBlenshapes ? &(holisticResultProto.face_blendshapes()) : nullptr;

  return [MPPHolisticLandmarkerResult
      holisticLandmarkerResultWithFaceLandmarksProto:holisticResultProto.face_landmarks()
                                faceBlendshapesProto:faceBlendshapesProto
                                  poseLandmarksProto:holisticResultProto.pose_landmarks()
                             poseWorldLandmarksProto:LandmarkList()
                           poseSegmentationMaskProto:{}
                              leftHandLandmarksProto:holisticResultProto.left_hand_landmarks()
                         leftHandWorldLandmarksProto:LandmarkList()
                             rightHandLandmarksProto:holisticResultProto.right_hand_landmarks()
                        rightHandWorldLandmarksProto:LandmarkList()
                             timestampInMilliseconds:0];
}

@end
