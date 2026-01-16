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

#import "mediapipe/tasks/ios/test/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+ProtobufHelpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/vision/gesture_recognizer/utils/sources/MPPGestureRecognizerResult+Helpers.h"

#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/tasks/cc/components/containers/proto/landmarks_detection_result.pb.h"
#include "mediapipe/tasks/ios/test/vision/utils/sources/parse_proto_utils.h"

namespace {
using ClassificationListProto = ::mediapipe::ClassificationList;
using ClassificationProto = ::mediapipe::Classification;
using LandmarksDetectionResultProto =
    ::mediapipe::tasks::containers::proto::LandmarksDetectionResult;
using ::mediapipe::tasks::ios::test::vision::utils::get_proto_from_pbtxt;
}  // anonymous namespace

@implementation MPPGestureRecognizerResult (ProtobufHelpers)

+ (MPPGestureRecognizerResult *)
    gestureRecognizerResultsFromProtobufFileWithName:(NSString *)fileName
                                        gestureLabel:(NSString *)gestureLabel
                               shouldRemoveZPosition:(BOOL)removeZPosition {
  LandmarksDetectionResultProto landmarkDetectionResultProto;

  if (!get_proto_from_pbtxt(fileName.cppString, landmarkDetectionResultProto).ok()) {
    return nil;
  }

  if (removeZPosition) {
    // Remove z position of landmarks, because they are not used in correctness testing. For video
    // or live stream mode, the z positions varies a lot during tracking from frame to frame.
    for (int i = 0; i < landmarkDetectionResultProto.landmarks().landmark().size(); i++) {
      auto &landmark = *landmarkDetectionResultProto.mutable_landmarks()->mutable_landmark(i);
      landmark.clear_z();
    }
  }

  ClassificationListProto gesturesProto;
  ClassificationProto *classificationProto = gesturesProto.add_classification();
  classificationProto->set_label([gestureLabel UTF8String]);

  return [MPPGestureRecognizerResult
      gestureRecognizerResultWithHandGesturesProto:{gesturesProto}
                                   handednessProto:{landmarkDetectionResultProto.classifications()}
                                handLandmarksProto:{landmarkDetectionResultProto.landmarks()}
                               worldLandmarksProto:{landmarkDetectionResultProto.world_landmarks()}
                           timestampInMilliseconds:0];
}

@end
