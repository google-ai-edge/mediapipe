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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/core/sources/MPPTaskPart.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPTextChunker.h"
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPVectorStore.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A configuration container class for initializing `MPPSemanticRetriever`.
 */
NS_SWIFT_NAME(SemanticRetrieverComponents)
@interface MPPSemanticRetrieverComponents : NSObject

/**
 * A custom chunker supplied by the user, or the default character-count sliding-window chunker if
 * none is provided.
 */
@property(nonatomic, strong) id<MPPTextChunker> chunker;

/**
 * A custom vector store storage backend.
 */
@property(nonatomic, strong) id<MPPVectorStore> vectorStore;

/**
 * List of embedding providers used to extract multi-modal embeddings.
 */
@property(nonatomic, copy) NSArray<id<MPPEmbeddingProvider>> *providers;

/**
 * Initializes the components with a vector store, an optional chunker, and list of providers.
 */
- (nullable instancetype)initWithVectorStore:(id<MPPVectorStore>)vectorStore
                                     chunker:(nullable id<MPPTextChunker>)chunker
                                   providers:(NSArray<id<MPPEmbeddingProvider>> *)providers
                                       error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
