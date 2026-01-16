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

#import "mediapipe/tasks/ios/vision/hand_landmarker/utils/sources/MPPHandLandmarkerOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace {
using CalculatorOptionsProto = mediapipe::CalculatorOptions;
using HandLandmarkerGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarkerGraphOptions;
using HandDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_detector::proto::HandDetectorGraphOptions;
using HandLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarksDetectorGraphOptions;
}  // namespace

@implementation MPPHandLandmarkerOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  HandLandmarkerGraphOptionsProto *handLandmarkerGraphOptionsProto =
      optionsProto
          ->MutableExtension(HandLandmarkerGraphOptionsProto::ext);
              handLandmarkerGraphOptionsProto->Clear();

  [self.baseOptions copyToProto:handLandmarkerGraphOptionsProto->mutable_base_options()
              withUseStreamMode:self.runningMode != MPPRunningModeImage];

  handLandmarkerGraphOptionsProto->set_min_tracking_confidence(self.minTrackingConfidence);

  HandDetectorGraphOptionsProto *handDetectorGraphOptionsProto =
      handLandmarkerGraphOptionsProto->mutable_hand_detector_graph_options();
  handDetectorGraphOptionsProto->set_num_hands(self.numHands);
  handDetectorGraphOptionsProto->set_min_detection_confidence(self.minHandDetectionConfidence);

  HandLandmarksDetectorGraphOptionsProto *handLandmarksDetectorGraphOptionsProto =
      handLandmarkerGraphOptionsProto->mutable_hand_landmarks_detector_graph_options();
  handLandmarksDetectorGraphOptionsProto->set_min_detection_confidence(
      self.minHandPresenceConfidence);
}

@end
