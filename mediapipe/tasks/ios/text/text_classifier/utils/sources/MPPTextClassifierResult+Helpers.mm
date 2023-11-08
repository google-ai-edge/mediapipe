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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"
#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

static const int kMicrosecondsPerMillisecond = 1000;

namespace {
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::Packet;
}  // namespace

#define int kMicrosecondsPerMillisecond = 1000;

@implementation MPPTextClassifierResult (Helpers)

+ (MPPTextClassifierResult *)textClassifierResultWithClassificationsPacket:(const Packet &)packet {
  MPPClassificationResult *classificationResult = [MPPClassificationResult
      classificationResultWithProto:packet.Get<ClassificationResultProto>()];

  return [[MPPTextClassifierResult alloc]
      initWithClassificationResult:classificationResult
           timestampInMilliseconds:(NSInteger)(packet.Timestamp().Value() /
                                               kMicrosecondsPerMillisecond)];
}

@end
