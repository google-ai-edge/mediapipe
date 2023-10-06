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

#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/processors/utils/sources/MPPClassifierOptions+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_classifier_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/gesture_recognizer/proto/hand_gesture_recognizer_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_detector/proto/hand_detector_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarker_graph_options.pb.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/proto/hand_landmarks_detector_graph_options.pb.h"

namespace {
using CalculatorOptionsProto = mediapipe::CalculatorOptions;
using GestureClassifierGraphOptionsProto =
    ::mediapipe::tasks::vision::gesture_recognizer::proto::GestureClassifierGraphOptions;
using GestureRecognizerGraphOptionsProto =
    ::mediapipe::tasks::vision::gesture_recognizer::proto::GestureRecognizerGraphOptions;
using HandGestureRecognizerGraphOptionsProto =
    ::mediapipe::tasks::vision::gesture_recognizer::proto::HandGestureRecognizerGraphOptions;
using HandLandmarkerGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarkerGraphOptions;
using HandDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_detector::proto::HandDetectorGraphOptions;
using HandLandmarksDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::hand_landmarker::proto::HandLandmarksDetectorGraphOptions;
using ClassifierOptionsProto = ::mediapipe::tasks::components::processors::proto::ClassifierOptions;
}  // namespace

@implementation MPPGestureRecognizerOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  GestureRecognizerGraphOptionsProto *gestureRecognizerGraphOptionsProto =
      optionsProto->MutableExtension(GestureRecognizerGraphOptionsProto::ext);
  gestureRecognizerGraphOptionsProto->Clear();

  [self.baseOptions copyToProto:gestureRecognizerGraphOptionsProto->mutable_base_options()
              withUseStreamMode:self.runningMode != MPPRunningModeImage];

  HandLandmarkerGraphOptionsProto *handLandmarkerGraphOptionsProto =
      gestureRecognizerGraphOptionsProto->mutable_hand_landmarker_graph_options();
  handLandmarkerGraphOptionsProto->Clear();
  handLandmarkerGraphOptionsProto->set_min_tracking_confidence(self.minTrackingConfidence);

  HandDetectorGraphOptionsProto *handDetectorGraphOptionsProto =
      handLandmarkerGraphOptionsProto->mutable_hand_detector_graph_options();
  handDetectorGraphOptionsProto->Clear();
  handDetectorGraphOptionsProto->set_num_hands(self.numHands);
  handDetectorGraphOptionsProto->set_min_detection_confidence(self.minHandDetectionConfidence);

  HandLandmarksDetectorGraphOptionsProto *handLandmarksDetectorGraphOptionsProto =
      handLandmarkerGraphOptionsProto->mutable_hand_landmarks_detector_graph_options();
  handLandmarksDetectorGraphOptionsProto->Clear();
  handLandmarksDetectorGraphOptionsProto->set_min_detection_confidence(
      self.minHandPresenceConfidence);

  HandGestureRecognizerGraphOptionsProto *handGestureRecognizerGraphOptionsProto =
      gestureRecognizerGraphOptionsProto->mutable_hand_gesture_recognizer_graph_options();

  if (self.cannedGesturesClassifierOptions) {
    GestureClassifierGraphOptionsProto *cannedGesturesClassifierOptionsProto =
        handGestureRecognizerGraphOptionsProto->mutable_canned_gesture_classifier_graph_options();
    cannedGesturesClassifierOptionsProto->Clear();
    [self.cannedGesturesClassifierOptions
        copyToProto:cannedGesturesClassifierOptionsProto->mutable_classifier_options()];
  }

  if (self.customGesturesClassifierOptions) {
    GestureClassifierGraphOptionsProto *customGesturesClassifierOptionsProto =
        handGestureRecognizerGraphOptionsProto->mutable_custom_gesture_classifier_graph_options();
    customGesturesClassifierOptionsProto->Clear();
    [self.customGesturesClassifierOptions
        copyToProto:customGesturesClassifierOptionsProto->mutable_classifier_options()];
  }
}

@end
