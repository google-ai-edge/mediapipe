// Copyright 2026 The MediaPipe Authors.
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

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief The result of a summarization operation.
 */
NS_SWIFT_NAME(TextSummarizerResult)
@interface MPPTextSummarizerResult : NSObject

/**
 * The generated summary text.
 */
@property(nonatomic, readonly) NSString *summary;

/**
 * Initializes a new `MPPTextSummarizerResult` with the given summary.
 *
 * @param summary The generated summary.
 *
 * @return An initialized `MPPTextSummarizerResult`.
 */
- (instancetype)initWithSummary:(NSString *)summary;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * @brief The result of a streaming summarization operation.
 */
NS_SWIFT_NAME(TextSummarizerStreamResult)
@interface MPPTextSummarizerStreamResult : NSObject

/**
 * The generated summary text chunk.
 */
@property(nonatomic, readonly) NSString *chunk;

/**
 * Whether the summarization is done.
 */
@property(nonatomic, readonly) BOOL done;

/**
 * Initializes a new `MPPTextSummarizerStreamResult` with the given chunk and done state.
 *
 * @param chunk The generated summary text chunk.
 * @param done Whether the summarization is done.
 *
 * @return An initialized `MPPTextSummarizerStreamResult`.
 */
- (instancetype)initWithChunk:(NSString *)chunk done:(BOOL)done;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
