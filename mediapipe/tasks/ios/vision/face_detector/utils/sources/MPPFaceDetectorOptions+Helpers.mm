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

#import "mediapipe/tasks/ios/vision/face_detector/utils/sources/MPPFaceDetectorOptions+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/core/utils/sources/MPPBaseOptions+Helpers.h"

#include "mediapipe/tasks/cc/vision/face_detector/proto/face_detector_graph_options.pb.h"

using CalculatorOptionsProto = ::mediapipe::CalculatorOptions;
using FaceDetectorGraphOptionsProto =
    ::mediapipe::tasks::vision::face_detector::proto::FaceDetectorGraphOptions;

@implementation MPPFaceDetectorOptions (Helpers)

- (void)copyToProto:(CalculatorOptionsProto *)optionsProto {
  FaceDetectorGraphOptionsProto *graphOptions =
      optionsProto->MutableExtension(FaceDetectorGraphOptionsProto::ext);

  graphOptions->Clear();

  [self.baseOptions copyToProto:graphOptions->mutable_base_options()];
  graphOptions->set_min_detection_confidence(self.minDetectionConfidence);
  graphOptions->set_min_suppression_threshold(self.minSuppressionThreshold);
}

@end
