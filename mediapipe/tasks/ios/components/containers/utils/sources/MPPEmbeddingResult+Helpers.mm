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

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbedding+Helpers.h"

#include "mediapipe/tasks/c/components/containers/embedding_result.h"

@implementation MPPEmbeddingResult (Helpers)

+ (MPPEmbeddingResult *)embeddingResultWithCResult:(const MpEmbeddingResult &)cResult {
  NSMutableArray<MPPEmbedding *> *embeddings =
      [NSMutableArray arrayWithCapacity:cResult.embeddings_count];
  for (uint32_t i = 0; i < cResult.embeddings_count; ++i) {
    [embeddings addObject:[MPPEmbedding embeddingWithCEmbedding:cResult.embeddings[i]]];
  }

  NSInteger timestampInMilliseconds = 0;
  if (cResult.has_timestamp_ms) {
    timestampInMilliseconds = (NSInteger)cResult.timestamp_ms;
  }

  return [[MPPEmbeddingResult alloc] initWithEmbeddings:embeddings
                                timestampInMilliseconds:timestampInMilliseconds];
}

@end
