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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

namespace {
static const int kMicrosecondsPerMillisecond = 1000;

using ::mediapipe::Image;
using LandmarkListProto = mediapipe::LandmarkList;
using NormalizedLandmarkListProto = mediapipe::NormalizedLandmarkList;
using ClassificationListProto = mediapipe::ClassificationList;
};  // namespace

#define NormalizedLandmarkListFromPacket(packet)            \
  packet.ValidateAsType<NormalizedLandmarkListProto>().ok() \
      ? packet.Get<NormalizedLandmarkListProto>()           \
      : NormalizedLandmarkListProto()

#define LandmarkListFromPacket(packet)                                              \
  packet.ValidateAsType<LandmarkListProto>().ok() ? packet.Get<LandmarkListProto>() \
                                                  : LandmarkListProto()

@implementation MPPHolisticLandmarkerResult (Helpers)

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultWithFaceLandmarksPacket:(const mediapipe::Packet &)faceLandmarksPacket
                              faceBlendshapesPacket:(const mediapipe::Packet &)faceBlendShapesPacket
                                poseLandmarksPacket:(const mediapipe::Packet &)poseLandmarksPacket
                           poseWorldLandmarksPacket:
                               (const mediapipe::Packet &)poseWorldLandmarksPacket
                         poseSegmentationMaskPacket:
                             (const mediapipe::Packet &)poseSegmentationMaskPacket
                            leftHandLandmarksPacket:
                                (const mediapipe::Packet &)leftHandLandmarksPacket
                       leftHandWorldLandmarksPacket:
                           (const mediapipe::Packet &)leftHandWorldLandmarksPacket
                           rightHandLandmarksPacket:
                               (const mediapipe::Packet &)rightHandLandmarksPacket
                      rightHandWorldLandmarksPacket:
                          (const mediapipe::Packet &)rightHandWorldLandmarksPacket {
  NSInteger timestampInMilliseconds =
      (NSInteger)(faceLandmarksPacket.Timestamp().Value() / kMicrosecondsPerMillisecond);

  const ClassificationListProto *faceBlendshapesProto =
      faceBlendShapesPacket.ValidateAsType<ClassificationListProto>().ok()
          ? &(faceBlendShapesPacket.Get<ClassificationListProto>())
          : nullptr;
  const Image *poseSegmentationMaskProto = poseSegmentationMaskPacket.ValidateAsType<Image>().ok()
                                               ? &(poseSegmentationMaskPacket.Get<Image>())
                                               : nullptr;

  return [MPPHolisticLandmarkerResult
      holisticLandmarkerResultWithFaceLandmarksProto:NormalizedLandmarkListFromPacket(
                                                         faceLandmarksPacket)
                                faceBlendshapesProto:faceBlendshapesProto
                                  poseLandmarksProto:NormalizedLandmarkListFromPacket(
                                                         poseLandmarksPacket)
                             poseWorldLandmarksProto:LandmarkListFromPacket(
                                                         poseWorldLandmarksPacket)
                           poseSegmentationMaskProto:poseSegmentationMaskProto
                              leftHandLandmarksProto:NormalizedLandmarkListFromPacket(
                                                         leftHandLandmarksPacket)
                         leftHandWorldLandmarksProto:LandmarkListFromPacket(
                                                         leftHandWorldLandmarksPacket)
                             rightHandLandmarksProto:NormalizedLandmarkListFromPacket(
                                                         rightHandLandmarksPacket)
                        rightHandWorldLandmarksProto:LandmarkListFromPacket(
                                                         rightHandWorldLandmarksPacket)
                             timestampInMilliseconds:timestampInMilliseconds];
}

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultWithFaceLandmarksProto:
        (const mediapipe::NormalizedLandmarkList &)faceLandmarksProto
                              faceBlendshapesProto:
                                  (const mediapipe::ClassificationList *)faceBlendshapesProto
                                poseLandmarksProto:
                                    (const mediapipe::NormalizedLandmarkList &)poseLandmarksProto
                           poseWorldLandmarksProto:
                               (const mediapipe::LandmarkList &)poseWorldLandmarksProto
                         poseSegmentationMaskProto:
                             (const mediapipe::Image *)poseSegmentationMaskProto
                            leftHandLandmarksProto:
                                (const mediapipe::NormalizedLandmarkList &)leftHandLandmarksProto
                       leftHandWorldLandmarksProto:
                           (const mediapipe::LandmarkList &)leftHandWorldLandmarksProto
                           rightHandLandmarksProto:
                               (const mediapipe::NormalizedLandmarkList &)rightHandLandmarksProto
                      rightHandWorldLandmarksProto:
                          (const mediapipe::LandmarkList &)rightHandWorldLandmarksProto
                           timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  NSArray<MPPNormalizedLandmark *> *faceLandmarks = [MPPHolisticLandmarkerResult
      normalizedLandmarksArrayFromNormalizedLandmarkListProto:faceLandmarksProto];

  NSArray<MPPNormalizedLandmark *> *poseLandmarks = [MPPHolisticLandmarkerResult
      normalizedLandmarksArrayFromNormalizedLandmarkListProto:poseLandmarksProto];
  NSArray<MPPLandmark *> *poseWorldLandmarks =
      [MPPHolisticLandmarkerResult landmarksArrayFromLandmarkListProto:poseWorldLandmarksProto];

  NSArray<MPPNormalizedLandmark *> *leftHandLandmarks = [MPPHolisticLandmarkerResult
      normalizedLandmarksArrayFromNormalizedLandmarkListProto:leftHandLandmarksProto];
  NSArray<MPPLandmark *> *leftHandWorldLandmarks =
      [MPPHolisticLandmarkerResult landmarksArrayFromLandmarkListProto:leftHandWorldLandmarksProto];

  NSArray<MPPNormalizedLandmark *> *rightHandLandmarks = [MPPHolisticLandmarkerResult
      normalizedLandmarksArrayFromNormalizedLandmarkListProto:rightHandLandmarksProto];
  NSArray<MPPLandmark *> *rightHandWorldLandmarks = [MPPHolisticLandmarkerResult
      landmarksArrayFromLandmarkListProto:rightHandWorldLandmarksProto];

  // Since the presence of faceBlendshapes and poseConfidenceMasks are optional, if they are not
  // present pass nil arrays to the result.
  MPPClassifications *faceBlendshapes;
  if (faceBlendshapesProto) {
    faceBlendshapes =
        [MPPClassifications classificationsWithClassificationListProto:*faceBlendshapesProto
                                                             headIndex:0
                                                              headName:@""];
  }

  MPPMask *poseConfidenceMask =
      poseSegmentationMaskProto
          ? [[MPPMask alloc]
                initWithFloat32Data:(float *)poseSegmentationMaskProto->GetImageFrameSharedPtr()
                                        .get()
                                        ->PixelData()
                              width:poseSegmentationMaskProto->width()
                             height:poseSegmentationMaskProto->height()
                         shouldCopy:YES]
          : nil;

  return [[MPPHolisticLandmarkerResult alloc] initWithFaceLandmarks:faceLandmarks
                                                    faceBlendshapes:faceBlendshapes
                                                      poseLandmarks:poseLandmarks
                                                 poseWorldLandmarks:poseWorldLandmarks
                                               poseSegmentationMask:poseConfidenceMask
                                                  leftHandLandmarks:leftHandLandmarks
                                             leftHandWorldLandmarks:leftHandWorldLandmarks
                                                 rightHandLandmarks:rightHandLandmarks
                                            rightHandWorldLandmarks:rightHandWorldLandmarks
                                            timestampInMilliseconds:timestampInMilliseconds];
}

+ (NSArray<MPPNormalizedLandmark *> *)normalizedLandmarksArrayFromNormalizedLandmarkListProto:
    (const NormalizedLandmarkListProto &)normalizedLandmarkListProto {
  NSMutableArray<MPPNormalizedLandmark *> *normalizedLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)normalizedLandmarkListProto.landmark().size()];
  for (const auto &normalizedLandmark : normalizedLandmarkListProto.landmark()) {
    [normalizedLandmarks
        addObject:[MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmark]];
  }

  return normalizedLandmarks;
}

+ (NSArray<MPPLandmark *> *)landmarksArrayFromLandmarkListProto:
    (const LandmarkListProto &)landmarkListProto {
  NSMutableArray<MPPLandmark *> *landmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];

  for (const auto &landmarkProto : landmarkListProto.landmark()) {
    MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
    [landmarks addObject:landmark];
  }

  return landmarks;
}

@end
