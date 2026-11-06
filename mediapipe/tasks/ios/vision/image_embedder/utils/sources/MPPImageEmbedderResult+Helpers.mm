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

#import "mediapipe/tasks/ios/vision/image_embedder/utils/sources/MPPImageEmbedderResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbeddingResult+Helpers.h"

#include "mediapipe/tasks/c/components/containers/embedding_result.h"
#include "mediapipe/tasks/c/vision/image_embedder/image_embedder.h"

@implementation MPPImageEmbedderResult (Helpers)

+ (nullable MPPImageEmbedderResult *)
    imageEmbedderResultWithCResult:(const ImageEmbedderResult &)cResult
           timestampInMilliseconds:(NSInteger)timestampInMilliseconds {
  MPPEmbeddingResult *embeddingResult = [MPPEmbeddingResult embeddingResultWithCResult:cResult];
  if (!embeddingResult) {
    return nil;
  }
  return [[MPPImageEmbedderResult alloc] initWithEmbeddingResult:embeddingResult
                                         timestampInMilliseconds:timestampInMilliseconds];
}

@end
