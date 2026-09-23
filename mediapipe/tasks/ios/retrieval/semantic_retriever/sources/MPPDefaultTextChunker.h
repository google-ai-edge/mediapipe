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
#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPTextChunker.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A default text chunking implementation using native C++ word-based chunking with overlap or
 * character sliding window.
 */
NS_SWIFT_NAME(DefaultTextChunker)
@interface MPPDefaultTextChunker : NSObject <MPPTextChunker>

/**
 * The chunk size to use for text chunking.
 * For character-based chunking, this is the character count per chunk.
 * For word-based chunking, this is the target word count per chunk.
 */
@property(nonatomic, readonly) NSInteger chunkSize;

/**
 * The chunk overlap to use for text chunking.
 * Used only in character-based chunking.
 */
@property(nonatomic, readonly) NSInteger chunkOverlap;

/**
 * The word threshold to use for text chunking.
 * Used only in word-based chunking.
 */
@property(nonatomic, readonly) NSInteger wordThreshold;

/**
 * Mode for text chunking.
 */
@property(nonatomic, readonly) MPPChunkingMode mode;

/**
 * Initializes the default text chunker with a chunk size, chunk overlap, word threshold, and
 * chunking mode.
 */
- (nullable instancetype)initWithChunkSize:(NSInteger)chunkSize
                              chunkOverlap:(NSInteger)chunkOverlap
                             wordThreshold:(NSInteger)wordThreshold
                                      mode:(MPPChunkingMode)mode
                                     error:(NSError **)error;

/**
 * Initializes the default text chunker with a chunk size, chunk overlap, and chunking mode.
 */
- (nullable instancetype)initWithChunkSize:(NSInteger)chunkSize
                              chunkOverlap:(NSInteger)chunkOverlap
                                      mode:(MPPChunkingMode)mode
                                     error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
