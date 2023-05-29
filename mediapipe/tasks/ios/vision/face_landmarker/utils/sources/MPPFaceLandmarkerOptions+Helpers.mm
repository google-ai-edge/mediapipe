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

#import "mediapipe/tasks/ios/vision/face_landmarker/utils/sources/MPPFaceLandmarkerOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/face_landmarker/proto/face_landmarks_detector_graph_options.pb.h"

using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using FaceDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions;
using FaceLandmarkerGraphOptionsProto =
    ::mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarkerGraphOptions;
using FaceLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_landmarker::proto::FaceLandmarksDetectorGraphOptions;

@implementation MPPFaceLandmarkerOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  FaceLandmarkerGraphOptionsProto *faceLandmarkerGraphOptions =
      optionsProto->MutableExtension(FaceLandmarkerGraphOptionsProto::ext);

  faceLandmarkerGraphOptions->Clear();

  [self.baseOptions copyToProto:faceLandmarkerGraphOptions->mutable_base_options()];
  faceLandmarkerGraphOptions->set_min_tracking_confidence(self.minTrackingConfidence);

  FaceLandmarksDetectorGraphOptionsProto *faceLandmarkerDetectorGraphOptions =
      faceLandmarkerGraphOptions->mutable_face_landmarks_detector_graph_options();
  faceLandmarkerDetectorGraphOptions->set_min_detection_confidence(self.minFacePresenceConfidence);

  FaceDetectorGraphOptionsProto *faceDetctorGraphOptions =
      faceLandmarkerGraphOptions->mutable_face_detector_graph_options();
  faceDetctorGraphOptions->set_num_faces(self.numFaces);
  faceDetctorGraphOptions->set_min_detection_confidence(self.minFaceDetectionConfidence);
}

@end
