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

#import "mediapipe/tasks/ios/audio/audio_classifier/utils/sources/MPPAudioClassifierResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPClassificationResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/classifications.pb.h"

static const int kMicrosecondsPerMillisecond = 1000;

namespace {
using ClassificationResultProto =
    ::mediapipe::tasks::components::containers::proto::ClassificationResult;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPAudioClassifierResult (Helpers)

+ (nullable MPPAudioClassifierResult *)audioClassifierResultWithClassificationsPacket:
    (const Packet &)packet {
  // Even if packet does not validate as the expected type, you can safely access the timestamp.
  NSInteger timestampInMilliseconds =
      (NSInteger)(packet.Timestamp().Value() / kMicrosecondsPerMillisecond);

  std::vector<ClassificationResultProto> cppClassificationResults;
  if (packet.ValidateAsType<ClassificationResultProto>().ok()) {
    // If `runningMode = .audioStream`, only a single `ClassificationResult` will be returned in the
    // result packet.
    cppClassificationResults.emplace_back(packet.Get<ClassificationResultProto>());
  } else if (packet.ValidateAsType<std::vector<ClassificationResultProto>>().ok()) {
    // If `runningMode = .audioStream`, a vector of timestamped `ClassificationResult`s will be
    // returned in the result packet.
    cppClassificationResults = packet.Get<std::vector<ClassificationResultProto>>();
  } else {
    // If packet does not contain protobuf of a type expected by the audio classifier.
    return [[MPPAudioClassifierResult alloc] initWithClassificationResults:@[]
                                                   timestampInMilliseconds:timestampInMilliseconds];
  }

  NSMutableArray<MPPClassificationResult *> *classificationResults =
      [NSMutableArray arrayWithCapacity:cppClassificationResults.size()];

  for (const auto &cppClassificationResult : cppClassificationResults) {
    MPPClassificationResult *classificationResult =
        [MPPClassificationResult classificationResultWithProto:cppClassificationResult];
    [classificationResults addObject:classificationResult];
  }

  return [[MPPAudioClassifierResult alloc] initWithClassificationResults:classificationResults
                                                 timestampInMilliseconds:timestampInMilliseconds];
}

@end
