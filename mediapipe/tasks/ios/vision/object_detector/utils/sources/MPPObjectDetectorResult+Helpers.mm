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

#import "mediapipe/tasks/ios/vision/object_detector/utils/sources/MPPObjectDetectorResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPDetection+Helpers.h"

namespace {
using DetectionProto = ::mediapipe::Detection;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPObjectDetectorResult (Helpers)

+ (nullable MPPObjectDetectorResult *)objectDetectorResultWithDetectionsPacket:
    (const Packet &)packet {

  NSInteger timestampInMilliseconds = (NSInteger)(packet.Timestamp().Value() /
                                                                      kMicrosecondsPerMillisecond);
  if (!packet.ValidateAsType<std::vector<DetectionProto>>().ok()) {
    return [[MPPObjectDetectorResult alloc] initWithDetections:@[]
                                  timestampInMilliseconds:timestampInMilliseconds];
  }

  const std::vector<DetectionProto> &detectionProtos = packet.Get<std::vector<DetectionProto>>();

  NSMutableArray<MPPDetection *> *detections =
      [NSMutableArray arrayWithCapacity:(NSUInteger)detectionProtos.size()];
  for (const auto &detectionProto : detectionProtos) {
    [detections addObject:[MPPDetection detectionWithProto:detectionProto]];
  }

  return
      [[MPPObjectDetectorResult alloc] initWithDetections:detections
                                  timestampInMilliseconds:timestampInMilliseconds];
}

@end
