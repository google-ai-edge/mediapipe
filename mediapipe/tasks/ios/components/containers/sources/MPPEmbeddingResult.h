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

#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbedding.h"

NS_ASSUME_NONNULL_BEGIN

/** Represents the embedding results of a model. Typically used as a result for embedding tasks. */
NS_SWIFT_NAME(EmbeddingResult)
@interface MPPEmbeddingResult : NSObject

/**
 * An Array of `MPPEmbedding` objects containing the embedding results for each head of the model.
 */
@property(nonatomic, readonly) NSArray<MPPEmbedding *> *embeddings;

/**
 * @brief The optional timestamp (in milliseconds) of the start of the chunk of data corresponding
 * to these results.
 * This is only used for embedding extraction on time series (e.g. audio embedder). In these use
 * cases, the amount of data to process might exceed the maximum size that the model can process. To
 * solve this, the input data is split into multiple chunks starting at different timestamps.
 */
@property(nonatomic, readonly) NSInteger timestampInMilliseconds;

/**
 * Initializes a new `MPPEmbedding` with the given array of embeddings and timestamp (in
 * milliseconds).
 *
 * @param embeddings An Array of `MPPEmbedding` objects containing the embedding results for each
 * head of the model.
 * @param timestampInMilliseconds The optional timestamp (in milliseconds) of the start of the chunk
 * of data corresponding to these results. Pass `0` if timestamp is absent.
 *
 * @return An instance of `MPPEmbeddingResult` initialized with the given array of embeddings and
 * timestamp (in milliseconds).
 */
- (instancetype)initWithEmbeddings:(NSArray<MPPEmbedding *> *)embeddings
           timestampInMilliseconds:(NSInteger)timestampInMilliseconds NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
