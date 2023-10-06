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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbeddingResult+Helpers.h"

#import "mediapipe/tasks/ios/common/utils/sources/NSString+Helpers.h"
#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbedding+Helpers.h"

namespace {
using EmbeddingResultProto = ::mediapipe::tasks::components::containers::proto::EmbeddingResult;
}

@implementation MPPEmbeddingResult (Helpers)

+ (MPPEmbeddingResult *)embeddingResultWithProto:
    (const EmbeddingResultProto &)embeddingResultProto {
  NSMutableArray *embeddings =
      [NSMutableArray arrayWithCapacity:(NSUInteger)embeddingResultProto.embeddings_size()];
  for (const auto &embeddingProto : embeddingResultProto.embeddings()) {
    [embeddings addObject:[MPPEmbedding embeddingWithProto:embeddingProto]];
  }

  NSInteger timestampInMilliseconds = 0;
  if (embeddingResultProto.has_timestamp_ms()) {
    timestampInMilliseconds = (NSInteger)embeddingResultProto.timestamp_ms();
  }

  return [[MPPEmbeddingResult alloc] initWithEmbeddings:embeddings
                                timestampInMilliseconds:timestampInMilliseconds];
}

@end
