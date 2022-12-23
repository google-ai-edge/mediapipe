// Copyright 2022 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/text/text_classifier/utils/sources/MPPTextClassifierResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

namespace {
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
}  // namespace

@implementation MPPTextClassifierResult (Helpers)

+ (MPPTextClassifierResult *)textClassifierResultWithProto:
    (const ClassificationResultProto &)classificationResultProto {
  long timeStamp;

  if (classificationResultProto.has_timestamp_ms()) {
    timeStamp = classificationResultProto.timestamp_ms();
  }

  MPPClassificationResult *classificationResult = [MPPClassificationResult classificationResultWithProto:classificationResultProto];

  return [[MPPTextClassifierResult alloc] initWithClassificationResult:classificationResult
                                                        timeStamp:timeStamp];
}

@end
