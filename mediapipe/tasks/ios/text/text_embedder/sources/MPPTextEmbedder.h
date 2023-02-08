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

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/text/text_embedder/sources/MPPTextEmbedderOptions.h"
#import "mediapipe/tasks/ios/text/text_embedder/sources/MPPTextEmbedderResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs embedding extraction on text.
 *
 * This API expects a TFLite model with (optional) [TFLite Model
 * Metadata](https://www.tensorflow.org/lite/convert/metadata").
 *
 * Metadata is required for models with int32 input tensors because it contains the input process
 * unit for the model's Tokenizer. No metadata is required for models with string input tensors.
 *
 * Input tensors:
 *  - Three input tensors `kTfLiteInt32` of shape `[batch_size x bert_max_seq_len]`
 *    representing the input ids, mask ids, and segment ids. This input signature requires
 *    a Bert Tokenizer process unit in the model metadata.
 *  - Or one input tensor `kTfLiteInt32` of shape `[batch_size x max_seq_len]` representing
 *    the input ids. This input signature requires a Regex Tokenizer process unit in the
 *    model metadata.
 *  - Or one input tensor (`kTfLiteString`) that is shapeless or has shape `[1]` containing
 *    the input string.
 *
 * At least one output tensor (`kTfLiteFloat32`/`kTfLiteUint8`) with shape `[1 x N]` where `N` is
 * the number of dimensions in the produced embeddings.
 */
NS_SWIFT_NAME(TextEmbedder)
@interface MPPTextEmbedder : NSObject

/**
 * Creates a new instance of `MPPTextEmbedder` from an absolute path to a TensorFlow Lite
 * model file stored locally on the device and the default `MPPTextEmbedderOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text embedder.
 *
 * @return A new instance of `MPPTextEmbedder` with the given model path. `nil` if there is an
 * error in initializing the text embedder.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPTextEmbedder` from the given `MPPTextEmbedderOptions`.
 *
 * @param options The options of type `MPPTextEmbedderOptions` to use for configuring the
 * `MPPTextEmbedder`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text embedder.
 *
 * @return A new instance of `MPPTextEmbedder` with the given options. `nil` if there is an
 * error in initializing the text embedder.
 */
- (nullable instancetype)initWithOptions:(MPPTextEmbedderOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs embedding extraction on the input text.
 *
 * @param text The `NSString` on which embedding extraction is to be performed.
 * @param error An optional error parameter populated when there is an error in performing
 * embedding extraction on the input text.
 *
 * @return  A `MPPTextEmbedderResult` object that contains a list of embeddings.
 */
- (nullable MPPTextEmbedderResult *)embedText:(NSString *)text
                                        error:(NSError **)error NS_SWIFT_NAME(embed(text:));

- (instancetype)init NS_UNAVAILABLE;

/**
 * Utility function to compute[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
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
+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error
    NS_SWIFT_NAME(cosineSimilarity(embedding1:embedding2:));

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
