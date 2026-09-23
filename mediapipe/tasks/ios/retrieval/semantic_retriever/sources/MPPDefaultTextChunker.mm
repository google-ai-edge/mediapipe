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

#import "mediapipe/tasks/ios/retrieval/semantic_retriever/sources/MPPDefaultTextChunker.h"

#import "mediapipe/tasks/ios/common/sources/MPPCommon.h"
#import "mediapipe/tasks/ios/common/utils/sources/MPPCommonUtils.h"

#include <string>
#include <vector>

#include "mediapipe/tasks/cc/text/utils/genai_utils.h"

@implementation MPPDefaultTextChunker

- (nullable instancetype)initWithChunkSize:(NSInteger)chunkSize
                              chunkOverlap:(NSInteger)chunkOverlap
                             wordThreshold:(NSInteger)wordThreshold
                                      mode:(MPPChunkingMode)mode
                                     error:(NSError **)error {
  if (chunkSize <= 0) {
    [MPPCommonUtils createCustomError:error
                             withCode:MPPTasksErrorCodeInvalidArgumentError
                          description:@"Chunk size must be positive."];
    return nil;
  }
  if (mode == MPPChunkingModeCharacter) {
    if (chunkOverlap < 0 || chunkOverlap >= chunkSize) {
      [MPPCommonUtils
          createCustomError:error
                   withCode:MPPTasksErrorCodeInvalidArgumentError
                description:@"Chunk overlap must be non-negative and less than chunk size."];
      return nil;
    }
  } else if (mode == MPPChunkingModeWord) {
    if (wordThreshold <= 0) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Word threshold must be positive."];
      return nil;
    }
    if (chunkSize > wordThreshold) {
      [MPPCommonUtils createCustomError:error
                               withCode:MPPTasksErrorCodeInvalidArgumentError
                            description:@"Chunk size (word chunk size) must be less than or equal "
                                        @"to word threshold."];
      return nil;
    }
  }

  self = [super init];
  if (self) {
    _chunkSize = chunkSize;
    _chunkOverlap = chunkOverlap;
    _wordThreshold = wordThreshold;
    _mode = mode;
  }
  return self;
}

- (nullable instancetype)initWithChunkSize:(NSInteger)chunkSize
                              chunkOverlap:(NSInteger)chunkOverlap
                                      mode:(MPPChunkingMode)mode
                                     error:(NSError **)error {
  if (mode == MPPChunkingModeWord) {
    NSInteger wordSize = chunkSize - chunkOverlap;
    return [self initWithChunkSize:wordSize
                      chunkOverlap:chunkOverlap
                     wordThreshold:chunkSize
                              mode:mode
                             error:error];
  } else {
    return [self initWithChunkSize:chunkSize
                      chunkOverlap:chunkOverlap
                     wordThreshold:0
                              mode:mode
                             error:error];
  }
}

- (nullable NSArray<NSString *> *)chunkText:(NSString *)text error:(NSError **)error {
  if (text.length == 0 ||
      [text stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
              .length == 0) {
    return @[];
  }

  std::vector<std::string> cppChunks;
  if (self.mode == MPPChunkingModeCharacter) {
    cppChunks = mediapipe::tasks::text::utils::ChunkTextByCountingCharacters(
        text.UTF8String, (int)self.chunkSize, (int)self.chunkOverlap);
  } else {
    int word_threshold = (int)self.wordThreshold;
    int word_chunk_size = (int)self.chunkSize;
    std::vector<mediapipe::tasks::text::utils::TextChunk> rawChunks =
        mediapipe::tasks::text::utils::ChunkText(text.UTF8String, word_threshold, word_chunk_size);
    cppChunks.reserve(rawChunks.size());
    for (const auto &chunk : rawChunks) {
      cppChunks.push_back(std::string(chunk.text));
    }
  }

  NSMutableArray<NSString *> *mutableChunks = [NSMutableArray arrayWithCapacity:cppChunks.size()];
  for (const auto &cppChunk : cppChunks) {
    [mutableChunks addObject:[NSString stringWithUTF8String:cppChunk.c_str()]];
  }
  return mutableChunks;
}

@end
