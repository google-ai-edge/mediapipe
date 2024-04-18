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

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/packet.h"

#import "mediapipe/tasks/ios/vision/holistic_landmarker/sources/MPPHolisticLandmarkerResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPHolisticLandmarkerResult (Helpers)

/**
 * Creates a `MPPHolisticLandmarkerResult` from face landmarks, face blend shapes, pose landmarks,
 * pose world landmarks, pose segmentation mask, left hand landmarks, left hand world landmarks,
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
 * @param poseSegmentationMaskPacket a MediaPipe packet wrapping a `mediapipe::Image`.
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
                         poseSegmentationMaskPacket:
                             (const mediapipe::Packet &)poseSegmentationMaskPacket
                            leftHandLandmarksPacket:
                                (const mediapipe::Packet &)leftHandLandmarksPacket
                       leftHandWorldLandmarksPacket:
                           (const mediapipe::Packet &)leftHandWorldLandmarksPacket
                           rightHandLandmarksPacket:
                               (const mediapipe::Packet &)rightHandLandmarksPacket
                      rightHandWorldLandmarksPacket:
                          (const mediapipe::Packet &)rightHandWorldLandmarksPacket;

/**
 * Creates a `MPPHolisticLandmarkerResult` from face landmarks, face blend shapes, pose landmarks,
 * pose world landmarks, pose segmentation mask, left hand landmarks, left hand world landmarks,
 * right hand landmarks, right hand world landmarks and timestamp in milliseconds.
 *
 * @param faceLandmarksProto A proto of type `mediapipe::NormalizedlandmarkList`.
 * @param faceBlendshapesProto A proto of type `mediapipe::ClassificationList`.
 * @param poseLandmarksProto A proto of type `mediapipe::NormalizedlandmarkList`.
 * @param poseWorldLandmarksProto A proto of type `mediapipe::LandmarkList`.
 * @param poseSegmentationMaskProto A proto of type `mediapipe::Image`.
 * @param leftHandLandmarksProto A proto of type `mediapipe::NormalizedlandmarkList`.
 * @param leftHandWorldLandmarksProto A proto of type `mediapipe::LandmarkList`.
 * @param rightHandLandmarksProto A proto of type `mediapipe::NormalizedlandmarkList`.
 * @param rightHandWorldLandmarksProto A proto of type `mediapipe::LandmarkList`.
 * @param timestampInMilliseconds The timestamp of the result.
 *
 * @return  A `MPPHolisticLandmarkerResult` object created from the given protos and timestamp in
 * milliseconds.
 */
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
                             (const ::mediapipe::Image *)poseSegmentationMaskProto
                            leftHandLandmarksProto:
                                (const mediapipe::NormalizedLandmarkList &)leftHandLandmarksProto
                       leftHandWorldLandmarksProto:
                           (const mediapipe::LandmarkList &)leftHandWorldLandmarksProto
                           rightHandLandmarksProto:
                               (const mediapipe::NormalizedLandmarkList &)rightHandLandmarksProto
                      rightHandWorldLandmarksProto:
                          (const mediapipe::LandmarkList &)rightHandWorldLandmarksProto
                           timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

NS_ASSUME_NONNULL_END
