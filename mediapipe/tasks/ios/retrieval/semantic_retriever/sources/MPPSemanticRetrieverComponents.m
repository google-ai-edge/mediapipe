// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPSemanticRetrieverComponents.h"

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPDefaultTextChunker.h"

static const NSInteger kDefaultChunkSize = 500;
static const NSInteger kDefaultChunkOverlap = 100;

@implementation MPPSemanticRetrieverComponents

- (nullable instancetype)initWithVectorStore:(id<MPPVectorStore>)vectorStore
                                     chunker:(nullable id<MPPTextChunker>)chunker
                                   providers:(NSArray<id<MPPEmbeddingProvider>> *)providers
                                       error:(NSError **)error {
  self = [super init];
  if (self) {
    _vectorStore = vectorStore;
    _providers = [providers copy];
    if (chunker) {
      _chunker = chunker;
    } else {
      _chunker = [[MPPDefaultTextChunker alloc] initWithChunkSize:kDefaultChunkSize
                                                     chunkOverlap:kDefaultChunkOverlap
                                                             mode:MPPChunkingModeCharacter
                                                            error:error];
      if (!_chunker) {
        return nil;
      }
    }
  }
  return self;
}

@end
