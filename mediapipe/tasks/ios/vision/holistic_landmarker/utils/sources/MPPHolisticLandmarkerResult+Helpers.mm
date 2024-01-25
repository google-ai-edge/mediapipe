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

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/utils/sources/MPPHolisticLandmarkerResult+Helpers.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPClassificationResult.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPLandmark.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

static constexpr int kMicrosecondsPerMillisecond = 1000;

using ::mediapipe::Packet;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ClassificationListProto = ::mediapipe::ClassificationList;
// using FaceGeometryProto = ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;

@implementation MPPHolisticLandmarkerResult (Helpers)

+ (MPPHolisticLandmarkerResult *)
     holisticLandmarkerResultWithFaceLandmarksPacket:(const mediapipe::Packet &)faceLandmarksPacket
                       faceWorldLandmarksPacket:(const mediapipe::Packet &)faceWorldLandmarksPacket
                       faceBlendshapesPacket:(const mediapipe::Packet &)faceBlendShapesPacket
                       poseLandmarksPacket:(const mediapipe::Packet &)poseLandmarksPacket
                       poseWorldLandmarksPacket:(const mediapipe::Packet &)poseWorldLandmarksPacket
                       poseSegmentationMasksPacket:(const mediapipe::Packet *)poseSegmentationMasksPacket
                       leftHandLandmarksPacket:(const mediapipe::Packet &)leftHandLandmarksPacket
                       leftHandWorldLandmarksPacket:(const mediapipe::Packet &)leftHandWorldLandmarksPacket
                       rightHandLandmarksPacket:(const mediapipe::Packet &)rightHandLandmarksPacket
                       rightHandWorldLandmarksPacket:(const mediapipe::Packet &)rightHandWorldLandmarksPacket {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *faceLandmarks;
  NSMutableArray<MPPClassifications *> *faceBlendshapes;
  NSMutableArray<MPPTransformMatrix *> *facialTransformationMatrixes;

  if (faceLandmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok()) {
    const std::vector<NormalizedLandmarkListProto> &faceNormalizedLandmarkListProtos =
        faceLandmarksPacket.Get<std::vector<NormalizedLandmarkListProto>>();
    faceLandmarks = [NSMutableArray arrayWithCapacity:(NSUInteger)faceNormalizedLandmarkListProtos.size()];
    for (const auto &faceNormalizedLandmarkListProto : faceNormalizedLandmarkListProtos) {
      NSMutableArray<MPPNormalizedLandmark *> *currentFaceLandmarks =
          [NSMutableArray arrayWithCapacity:(NSUInteger)faceNormalizedLandmarkListProto.landmark_size()];
      for (const auto &faceNormalizedLandmark : faceNormalizedLandmarkListProto.landmark()) {
        [currentFaceLandmarks
            addObject:[MPPNormalizedLandmark normalizedLandmarkWithProto:faceNormalizedLandmark]];
      }
      [faceLandmarks addObject:currentFaceLandmarks];
    }
  } else {
    faceLandmarks = [NSMutableArray arrayWithCapacity:0];
  }

  if (blendshapesPacket.ValidateAsType<std::vector<ClassificationListProto>>().ok()) {
    const std::vector<ClassificationListProto> &classificationListProtos =
        blendshapesPacket.Get<std::vector<ClassificationListProto>>();
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

  for (const auto &landmarkListProto : landmarksProto) {
    NSMutableArray<MPPNormalizedLandmark *> *landmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark().size()];
    for (const auto &normalizedLandmarkProto : landmarkListProto.landmark()) {
      MPPNormalizedLandmark *normalizedLandmark =
          [MPPNormalizedLandmark normalizedLandmarkWithProto:normalizedLandmarkProto];
      [landmarks addObject:normalizedLandmark];
    }
    [multiplePoseLandmarks addObject:landmarks];
  }

  NSMutableArray<NSMutableArray<MPPLandmark *> *> *multiplePoseWorldLandmarks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarksProto.size()];

  for (const auto &worldLandmarkListProto : worldLandmarksProto) {
    NSMutableArray<MPPLandmark *> *worldLandmarks =
        [NSMutableArray arrayWithCapacity:(NSUInteger)worldLandmarkListProto.landmark().size()];
    for (const auto &landmarkProto : worldLandmarkListProto.landmark()) {
      MPPLandmark *landmark = [MPPLandmark landmarkWithProto:landmarkProto];
      [worldLandmarks addObject:landmark];
    }
    [multiplePoseWorldLandmarks addObject:worldLandmarks];
  }

  if (!segmentationMasks) {
    return [[MPPPoseLandmarkerResult alloc] initWithLandmarks:multiplePoseLandmarks
                                               worldLandmarks:multiplePoseWorldLandmarks
                                            segmentationMasks:nil
                                      timestampInMilliseconds:timestampInMilliseconds];
  }
  NSMutableArray<MPPMask *> *confidenceMasks =
      [NSMutableArray arrayWithCapacity:(NSUInteger)segmentationMasks->size()];

  for (const auto &segmentationMask : *segmentationMasks) {
    [confidenceMasks addObject:[[MPPMask alloc] initWithFloat32Data:(float *)segmentationMask
                                                                        .GetImageFrameSharedPtr()
                                                                        .get()
                                                                        ->PixelData()
                                                              width:segmentationMask.width()
                                                             height:segmentationMask.height()
                                                         /** Always deep copy */
                                                         shouldCopy:YES]];

  return [[MPPFaceLandmarkerResult alloc]
             initWithFaceLandmarks:faceLandmarks
                   faceBlendshapes:faceBlendshapes
      facialTransformationMatrixes:facialTransformationMatrixes
           timestampInMilliseconds:(NSInteger)(landmarksPacket.Timestamp().Value() /
                                               kMicrosecondsPerMillisecond)];
}

@end
