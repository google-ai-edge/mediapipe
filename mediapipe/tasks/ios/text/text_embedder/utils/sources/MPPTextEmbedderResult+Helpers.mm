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

#import "mediapipe/tasks/ios/text/text_embedder/utils/sources/MPPTextEmbedderResult+Helpers.h"

#import "mediapipe/tasks/ios/components/containers/utils/sources/MPPEmbeddingResult+Helpers.h"

@implementation MPPTextEmbedderResult (Helpers)

+ (MPPTextEmbedderResult *)textEmbedderResultWithCResult:(const MpTextEmbedderResult &)cResult {
  MPPEmbeddingResult *embeddingResult = [MPPEmbeddingResult embeddingResultWithCResult:cResult];

  return [[MPPTextEmbedderResult alloc]
      initWithEmbeddingResult:embeddingResult
      timestampInMilliseconds:(NSInteger)(cResult.has_timestamp_ms ? cResult.timestamp_ms : 0)];
}

@end
