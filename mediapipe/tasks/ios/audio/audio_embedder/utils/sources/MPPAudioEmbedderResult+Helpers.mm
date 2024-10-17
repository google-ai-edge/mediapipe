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

#import "mediapipe/tasks/ios/audio/audio_embedder/utils/sources/MPPAudioEmbedderResult+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbeddingResult+Helpers.h"

#include "mediapipe/tasks/cc/components/containers/proto/embeddings.pb.h"

static const int kMicrosecondsPerMillisecond = 1000;

namespace {
using EmbeddingResultProto = ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
using ::mediapipe::Packet;
}  // namespace

@implementation MPPAudioEmbedderResult (Helpers)

+ (MPPAudioEmbedderResult *)audioEmbedderResultWithEmbeddingResultPacket:(const Packet &)packet {
  // Even if packet does not validate as the expected type, you can safely access the timestamp.
  NSInteger timestampInMilliseconds =
      (NSInteger)(packet.Timestamp().Value() / kMicrosecondsPerMillisecond);

  std::vector<EmbeddingResultProto> cppEmbeddingResults;
  if (packet.ValidateAsType<EmbeddingResultProto>().ok()) {
    // If `runningMode = .audioStream`, only a single `EmbeddingResult` will be returned in the
    // result packet.
    cppEmbeddingResults.emplace_back(packet.Get<EmbeddingResultProto>());
  } else if (packet.ValidateAsType<std::vector<EmbeddingResultProto>>().ok()) {
    // If `runningMode = .audioStream`, a vector of timestamped `EmbeddingResult`s will be
    // returned in the result packet.
    cppEmbeddingResults = packet.Get<std::vector<EmbeddingResultProto>>();
  } else {
    // If packet does not contain protobuf of a type expected by the audio embedder.
    return [[MPPAudioEmbedderResult alloc] initWithEmbeddingResults:@[]
                                            timestampInMilliseconds:timestampInMilliseconds];
  }

  NSMutableArray<MPPEmbeddingResult *> *embeddingResults =
      [NSMutableArray arrayWithCapacity:cppEmbeddingResults.size()];

  for (const auto &cppEmbeddingResult : cppEmbeddingResults) {
    MPPEmbeddingResult *embeddingResult =
        [MPPEmbeddingResult embeddingResultWithProto:cppEmbeddingResult];
    [embeddingResults addObject:embeddingResult];
  }

  return [[MPPAudioEmbedderResult alloc] initWithEmbeddingResults:embeddingResults
                                          timestampInMilliseconds:timestampInMilliseconds];
}

@end
