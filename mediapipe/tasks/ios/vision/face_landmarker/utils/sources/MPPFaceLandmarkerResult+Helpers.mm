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

#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerResult+Helpers.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/matrix_data.pb.h"
#include "mediapipe/tasks/cc/vision/face_geometry/proto/face_geometry.pb.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPClassificationResult.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPLandmark.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPLandmark+Helpers.h"

static constexpr int kMicrosecondsPerMillisecond = 1000;

using ::mediapipe::Packet;
using NormalizedLandmarkListProto = ::mediapipe::NormalizedLandmarkList;
using ClassificationListProto = ::mediapipe::ClassificationList;
using FaceGeometryProto = ::mediapipe::tasks::vision::face_geometry::proto::FaceGeometry;

@implementation MPPFaceLandmarkerResult (Helpers)

+ (MPPFaceLandmarkerResult *)
    faceLandmarkerResultWithLandmarksPacket:(const Packet &)landmarksPacket
                          blendshapesPacket:(const Packet &)blendshapesPacket
               transformationMatrixesPacket:(const Packet &)transformationMatrixesPacket {
  NSMutableArray<NSArray<MPPNormalizedLandmark *> *> *faceLandmarks;
  NSMutableArray<MPPClassifications *> *faceBlendshapes;
  NSMutableArray<MPPTransformMatrix *> *facialTransformationMatrixes;

  if (landmarksPacket.ValidateAsType<std::vector<NormalizedLandmarkListProto>>().ok()) {
    const std::vector<NormalizedLandmarkListProto> &landmarkListProtos =
        landmarksPacket.Get<std::vector<NormalizedLandmarkListProto>>();
    faceLandmarks = [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProtos.size()];
    for (const auto &landmarkListProto : landmarkListProtos) {
      NSMutableArray<MPPNormalizedLandmark *> *currentFaceLandmarks =
          [NSMutableArray arrayWithCapacity:(NSUInteger)landmarkListProto.landmark_size()];
      for (const auto &landmarkProto : landmarkListProto.landmark()) {
        [currentFaceLandmarks
            addObject:[MPPNormalizedLandmark normalizedLandmarkWithProto:landmarkProto]];
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
  } else {
    faceBlendshapes = [NSMutableArray arrayWithCapacity:0];
  }

  if (transformationMatrixesPacket.ValidateAsType<std::vector<FaceGeometryProto>>().ok()) {
    const std::vector<FaceGeometryProto> &geometryProtos =
        transformationMatrixesPacket.Get<std::vector<FaceGeometryProto>>();
    facialTransformationMatrixes =
        [NSMutableArray arrayWithCapacity:(NSUInteger)geometryProtos.size()];
    for (const auto &geometryProto : geometryProtos) {
      MPPTransformMatrix *transformMatrix = [[MPPTransformMatrix alloc]
          initWithData:geometryProto.pose_transform_matrix().packed_data().data()
                  rows:geometryProto.pose_transform_matrix().rows()
               columns:geometryProto.pose_transform_matrix().cols()];
      [facialTransformationMatrixes addObject:transformMatrix];
    }
  } else {
    facialTransformationMatrixes = [NSMutableArray arrayWithCapacity:0];
  }

  return [[MPPFaceLandmarkerResult alloc]
             initWithFaceLandmarks:faceLandmarks
                   faceBlendshapes:faceBlendshapes
      facialTransformationMatrixes:facialTransformationMatrixes
           timestampInMilliseconds:(NSInteger)(landmarksPacket.Timestamp().Value() /
                                               kMicrosecondsPerMillisecond)];
}

@end
