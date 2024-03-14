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

@implementation MPPHolisticLandmarkerResult (Helpers)

+ (MPPHolisticLandmarkerResult *)
    holisticLandmarkerResultWithFaceLandmarksPacket:(const mediapipe::Packet &)faceLandmarksPacket
                              faceBlendshapesPacket:(const mediapipe::Packet &)faceBlendShapesPacket
                                poseLandmarksPacket:(const mediapipe::Packet &)poseLandmarksPacket
                           poseWorldLandmarksPacket:
                               (const mediapipe::Packet &)poseWorldLandmarksPacket
                        poseSegmentationMasksPacket:
                            (const mediapipe::Packet *)poseSegmentationMasksPacket
                            leftHandLandmarksPacket:
                                (const mediapipe::Packet &)leftHandLandmarksPacket
                       leftHandWorldLandmarksPacket:
                           (const mediapipe::Packet &)leftHandWorldLandmarksPacket
                           rightHandLandmarksPacket:
                               (const mediapipe::Packet &)rightHandLandmarksPacket
                      rightHandWorldLandmarksPacket:
                          (const mediapipe::Packet &)rightHandWorldLandmarksPacket {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *faceLandmarks =
      [MPPHolisticLandmarkerResult normalizedLandmarksArrayFromPacket:faceLandmarksPacket];

  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *poseLandmarks =
      [MPPHolisticLandmarkerResult normalizedLandmarksArrayFromPacket:poseLandmarksPacket];
  NSMutableArray<NSArray<MPPLandmark *> *> *poseWorldLandmarks =
      [MPPHolisticLandmarkerResult landmarksArrayFromPacket:poseWorldLandmarksPacket];

  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *leftHandLandmarks =
      [MPPHolisticLandmarkerResult normalizedLandmarksArrayFromPacket:leftHandLandmarksPacket];
  NSMutableArray<NSArray<MPPLandmark *> *> *leftHandWorldLandmarks =
      [MPPHolisticLandmarkerResult landmarksArrayFromPacket:leftHandWorldLandmarksPacket];

  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *rightHandLandmarks =
      [MPPHolisticLandmarkerResult normalizedLandmarksArrayFromPacket:rightHandLandmarksPacket];
  NSMutableArray<NSArray<MPPLandmark *> *> *rightHandWorldLandmarks =
      [MPPHolisticLandmarkerResult landmarksArrayFromPacket:rightHandWorldLandmarksPacket];

  // Since the presence of faceBlendshapes and poseConfidenceMasks are optional, if they are not
  // present pass nil arrays to the result.
  NSMutableArray<MPPClassifications *> *faceBlendshapes;
  if (faceBlendShapesPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok()) {
    const std::vector<ClassificationListProto> &classificationListProtos =
        faceBlendShapesPacket.Get<std::vector<ClassificationListProto>>();
    faceBlendshapes =
        [NSMutableArray arrayWithCapacity:(NSUInteger)classificationListProtos.size()];
    for (const auto &classificationListProto : classificationListProtos) {
      [faceBlendshapes
          addObject:[MPPClassifications
                        classificationsWithClassificationListProto:classificationListProto
                                                         headIndex:0
                                                          headName:@""]];
    }
  }

  NSMutableArray<MPPMask *> *poseConfidenceMasks;
  if (poseSegmentationMasksPacket->ValidateAsType<std::vector<Image>>().ok()) {
    std::vector<Image> cppConfidenceMasks = poseSegmentationMasksPacket->Get<std::vector<Image>>();
    poseConfidenceMasks = [NSMutableArray arrayWithCapacity:(NSUInteger)cppConfidenceMasks.size()];

    for (const auto &confidenceMask : cppConfidenceMasks) {
      [poseConfidenceMasks
          addObject:[[MPPMask alloc]
                        initWithFloat32Data:(float *)confidenceMask.GetImageFrameSharedPtr()
                                                .get()
                                                ->PixelData()
                                      width:confidenceMask.width()
                                     height:confidenceMask.height()
                                 shouldCopy:YES]];
    }
  }

  return [[MPPHolisticLandmarkerResult alloc]
        initWithFaceLandmarks:faceLandmarks
              faceBlendshapes:faceBlendshapes
                poseLandmarks:poseLandmarks
           poseWorldLandmarks:poseWorldLandmarks
        poseSegmentationMasks:poseConfidenceMasks
            leftHandLandmarks:leftHandLandmarks
       leftHandWorldLandmarks:leftHandWorldLandmarks
           rightHandLandmarks:rightHandLandmarks
      rightHandWorldLandmarks:rightHandWorldLandmarks
      timestampInMilliseconds:(NSInteger)(faceLandmarksPacket.Timestamp().Value() /
                                          kMicrosecondsPerMillisecond)];
}

+ (NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *)normalizedLandmarksArrayFromPacket:
    (const mediapipe::Packet &)normalizedLandmarksPacket {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *normalizedLandmarks =
      [MPPHolisticLandmarkerResult
          nullableNormalizedLandmarksArrayFromPacket:normalizedLandmarksPacket];

  return normalizedLandmarks ? normalizedLandmarks : [NSMutableArray arrayWithCapacity:0];
}

+ (NSMutableArray<NSArray<MPPLandmark *> *> *)landmarksArrayFromPacket:
    (const mediapipe::Packet &)landmarksPacket {
  NSMutableArray<NSArray<MPPLandmark *> *> *landmarks =
      [MPPHolisticLandmarkerResult nullableLandmarksArrayFromPacket:landmarksPacket];

  return landmarks ? landmarks : [NSMutableArray arrayWithCapacity:0];
}

+ (NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *)nullableNormalizedLandmarksArrayFromPacket:
    (const mediapipe::Packet &)normalizedLandmarksPacket {
  if (!normalizedLandmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok()) {
    return nil;
  }

  const std::vector<NormalizedLandmarkListProto> &normalizedLandmarkListProtos =
      normalizedLandmarksPacket.Get<std::vector<NormalizedLandmarkListProto>>();

  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *multipleNormalizedLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)normalizedLandmarkListProtos.size()];
  for (const auto &normalizedLandmarkListProto : normalizedLandmarkListProtos) {
    NSMutableArray<MPPNormalizedLandmark *> *normalizedLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)normalizedLandmarkListProto.landmark_size()];
    for (const auto &normalizedLandmark : normalizedLandmarkListProto.landmark()) {
      [normalizedLandmarks
          addObject:[MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmark]];
    }
    [multipleNormalizedLandmarks addObject:normalizedLandmarks];
  }

  return multipleNormalizedLandmarks;
}

+ (NSMutableArray<NSArray<MPPLandmark *> *> *)nullableLandmarksArrayFromPacket:
    (const mediapipe::Packet &)landmarksPacket {
  if (!landmarksPacket.ValidateAsType<std::vector<LandmarkListProto>>().ok()) {
    return nil;
  }

  const std::vector<LandmarkListProto> &landmarkListProtos =
      landmarksPacket.Get<std::vector<LandmarkListProto>>();

  NSMutableArray<NSArray<MPPLandmark *> *> *multipleLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProtos.size()];

  for (const auto &landmarkListProto : landmarkListProtos) {
    NSMutableArray<MPPLandmark *> *landmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];
    for (const auto &landmarkProto : landmarkListProto.landmark()) {
      MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
      [landmarks addObject:landmark];
    }
    [multipleLandmarks addObject:landmarks];
  }

  return multipleLandmarks;
}

@end
