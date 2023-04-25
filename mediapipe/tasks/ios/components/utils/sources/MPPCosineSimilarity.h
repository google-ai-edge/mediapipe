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

/** Utility class for computing cosine similarity between `MPPEmbedding` objects. */
NS_SWIFT_NAME(CosineSimilarity)

@interface MPPCosineSimilarity : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

/** Utility function to compute[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
 * between two `MPPEmbedding` objects.
 *
 * @param embedding1 One of the two `MPPEmbedding`s between whom cosine similarity is to be
 * computed.
 * @param embedding2 One of the two `MPPEmbedding`s between whom cosine similarity is to be
 * computed.
 * @param error An optional error parameter populated when there is an error in calculating cosine
 * similarity between two embeddings.
 *
 * @return An `NSNumber` which holds the cosine similarity of type `double`.
 */
+ (nullable NSNumber *)computeBetweenEmbedding1:(MPPEmbedding *)embedding1
                                  andEmbedding2:(MPPEmbedding *)embedding2
                                          error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
