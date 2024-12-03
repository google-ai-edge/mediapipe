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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierResult.h"

#include "mediapipe/framework/packet.h"

NS_ASSUME_NONNULL_BEGIN

@interface MPPAudioClassifierResult (Helpers)

/**
 * Creates an `MPPAudioClassifierResult` from a MediaPipe packet containing a
 * std::vector<ClassificationResult>.
 *
 * @param packet a MediaPipe packet wrapping a ClassificationResultProto.
 *
 * @return  An `MPPAudioClassifierResult` object that contains a list of audio classifications.
 */
+ (nullable MPPAudioClassifierResult *)audioClassifierResultWithClassificationsPacket:
    (const mediapipe::Packet &)packet;

@end

NS_ASSUME_NONNULL_END
