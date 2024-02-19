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

#ifndef __cplusplus
#error "This file requires Objective-C++."
#endif  // __cplusplus

#include "mediapipe/framework/packet.h"
#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPHolisticLandmarkerResult (Helpers)

/**
 * Creates an `MPPHolisticLandmarkerResult` from face landmarks, face blend shapes, pose landmarks,
 * pose world landmarks, pose segmentation masks, left hand landmarks, left hand world landmarks,
 * right hand landmarks and right hand world landmarks packets.
 *
 * @param faceLandmarksPacket A MediaPipe packet wrapping a
 * `std::vector<mediapipe::NormalizedLandmarkList>`.
 * @param faceBlendShapesPacket A MediaPipe packet wrapping a
 * `std::vector<mediapipe::ClassificationList>`.
 * @param poseLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::NormalizedlandmarkList>`.
 * @param poseWorldLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::LandmarkList>`.
 * @param poseSegmentationMasksPacket a MediaPipe packet wrapping a `std::vector<mediapipe::Image>`.
 * @param leftHandLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::NormalizedlandmarkList>`.
 * @param leftHandWorldLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::LandmarkList>`.
 * @param rightHandLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::NormalizedlandmarkList>`.
 * @param rightHandWorldLandmarksPacket a MediaPipe packet wrapping a
 * `std::vector<mediapipe::LandmarkList>`.
 *
 * @return  An `MPPHolisticLandmarkerResult` object that contains the holistic landmark detection
 * results.
 */
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
                          (const mediapipe::Packet &)rightHandWorldLandmarksPacket;

@end
NS_ASSUME_NONNULL_END
