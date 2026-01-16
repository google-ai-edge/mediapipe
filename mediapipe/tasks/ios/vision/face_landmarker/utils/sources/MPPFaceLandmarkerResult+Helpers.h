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

#ifndef __cplusplus
#error "This file requires Objective-C++."
#endif  // __cplusplus

#include "mediapipe/framework/packet.h"
#import "mediapipe/tasks/ios/vision/face_landmarker/sources/MPPFaceLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPFaceLandmarkerResult (Helpers)

/**
 * Creates an `MPPFaceLandmarkerResult` from the MediaPipe packets containing the results of the
 * FaceLandmarker.
 *
 * @param landmarksPacket a MediaPipe packet wrapping a `std::vector<NormalizedLandmarkProto>`.
 * @param blendshapesPacket a MediaPipe packet wrapping a `std::vector<ClassificationProto>`.
 * @param transformationMatrixesPacket a MediaPipe packet wrapping a
 * `std::vector<FaceGeometryProto>`.
 *
 * @return An `MPPFaceLandmarkerResult` object that contains the contents of the provided packets.
 */
+ (MPPFaceLandmarkerResult *)
    faceLandmarkerResultWithLandmarksPacket:(const ::mediapipe::Packet &)landmarksPacket
                          blendshapesPacket:(const ::mediapipe::Packet &)blendshapesPacket
               transformationMatrixesPacket:
                   (const ::mediapipe::Packet &)transformationMatrixesPacket;
@end

NS_ASSUME_NONNULL_END
