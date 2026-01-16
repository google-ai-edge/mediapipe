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
#import "mediapipe/tasks/ios/vision/image_classifier/utils/sources/MPPImageClassifierResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

static const int kMicrosecondsPerMillisecond = 1000;

namespace {
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPImageClassifierResult (Helpers)

+ (nullable MPPImageClassifierResult *)imageClassifierResultWithClassificationsPacket:
    (const Packet &)packet {
  // Even if packet does not validate as the expected type, you can safely access the timestamp.
  NSInteger timestampInMilliSeconds =
      (NSInteger)(packet.Timestamp().Value() / kMicrosecondsPerMillisecond);

  if (!packet.ValidateAsType<ClassificationResultProto>().ok()) {
    // MPPClassificationResult's timestamp is populated from timestamp `ClassificationResultProto`'s
    // timestamp_ms(). It is 0 since the packet can't be validated as a `ClassificationResultProto`.
    return [[MPPImageClassifierResult alloc]
        initWithClassificationResult:[[MPPClassificationResult alloc] initWithClassifications:@[]
                                                                      timestampInMilliseconds:0]
             timestampInMilliseconds:timestampInMilliSeconds];
  }

  MPPClassificationResult *classificationResult = [MPPClassificationResult
      classificationResultWithProto:packet.Get<ClassificationResultProto>()];

  return [[MPPImageClassifierResult alloc]
      initWithClassificationResult:classificationResult
           timestampInMilliseconds:(NSInteger)(packet.Timestamp().Value() /
                                               kMicrosecondsPerMillisecond)];
}

@end
