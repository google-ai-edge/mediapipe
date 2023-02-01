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

NS_ASSUME_NONNULL_BEGIN

/**
 * Represents the embedding for a given embedder head. Typically used in embedding tasks.
 *
 * One and only one of the two 'floatEmbedding' and 'quantizedEmbedding' will contain data, based on
 * whether or not the embedder was configured to perform scala quantization.
 */
NS_SWIFT_NAME(Embedding)
@interface MPPEmbedding : NSObject

/**
 * @brief The embedding represented as an `NSArray` of `Float` values.
 * Empty if the embedder was configured to perform scalar quantization.
 */
@property(nonatomic, readonly, nullable) NSArray<NSNumber *> *floatEmbedding;

/**
 * @brief The embedding represented as an `NSArray` of `UInt8` values.
 * Empty if the embedder was not configured to perform scalar quantization.
 */
@property(nonatomic, readonly, nullable) NSArray<NSNumber *> *quantizedEmbedding;

/** The index of the embedder head these entries refer to. This is useful for multi-head models. */
@property(nonatomic, readonly) NSInteger headIndex;

/** The optional name of the embedder head, which is the corresponding tensor metadata name. */
@property(nonatomic, readonly, nullable) NSString *headName;

/**
 * Initializes a new `MPPEmbedding` with the given float embedding, quantized embedding, head index
 * and head name.
 *
 * @param floatEmbedding The optional Floating-point embedding.
 * @param quantizedEmbedding The optional Quantized embedding.
 * @param headIndex The index of the embedder head.
 * @param headName The optional name of the embedder head.
 *
 * @return An instance of `MPPEmbedding` initialized with the given float embedding, quantized
 * embedding, head index and head name.
 */
- (instancetype)initWithFloatEmbedding:(nullable NSArray<NSNumber *> *)floatEmbedding
                    quantizedEmbedding:(nullable NSArray<NSNumber *> *)quantizedEmbedding
                             headIndex:(NSInteger)headIndex
                              headName:(nullable NSString *)headName NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
