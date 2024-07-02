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

  if (!packet.ValidateAsType<EmbeddingResultProto>().ok()) {
    // MPPAudioEmbedderResult's timestamp is populated from timestamp `EmbeddingResultProto`'s
    // timestamp_ms(). It is 0 since the packet can't be validated as a `EmbeddingResultProto`.
    return [[MPPAudioEmbedderResult alloc] initWithEmbeddingResults:nil
                                           timestampInMilliseconds:timestampInMilliseconds];
  }

  std::vector<EmbeddingResultProto> cppEmbeddingResultProtos =
      packet.Get<std::vector<EmbeddingResultProto>>();

  NSMutableArray<MPPEmbeddingResult *> *embeddingResults =
      [NSMutableArray arrayWithCapacity:cppEmbeddingResultProtos.size()];

  for (const auto &cppEmbeddingResultProto : cppEmbeddingResultProtos) {
    MPPEmbeddingResult *embeddingResult =
        [MPPEmbeddingResult embeddingResultWithProto:cppEmbeddingResultProto];
    [embeddingResults addObject:embeddingResult];
  }

  return [[MPPAudioEmbedderResult alloc] initWithEmbeddingResults:embeddingResults
                                          timestampInMilliseconds:timestampInMilliseconds];
}

@end
