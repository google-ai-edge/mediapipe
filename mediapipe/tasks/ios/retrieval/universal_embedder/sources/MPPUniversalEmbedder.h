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

#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"
#import "mediapipe/tasks/ios/components/containers/sources/MPPEmbeddingResult.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskPart.h"
#import "mediapipe/tasks/ios/retrieval/universal_embedder/sources/MPPUniversalEmbedderOptions.h"
#import "mediapipe/tasks/ios/vision/core/sources/MPPImage.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Performs multi-modal embedding extraction on text, image, audio, or custom content blocks.
 *
 * This API wraps the native LiteRT-LM EmbeddingEngine directly, providing real on-device
 * multimodal embeddings.
 */
NS_SWIFT_NAME(UniversalEmbedder)
@interface MPPUniversalEmbedder : NSObject <MPPEmbeddingProvider>

/**
 * Creates a new instance of `MPPUniversalEmbedder` from an absolute path to a model file stored
 * locally on the device and the default options.
 *
 * @param modelPath An absolute path to a model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * universal embedder.
 * @return A new instance of `MPPUniversalEmbedder` with the given model path. `nil` if there is an
 * error.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPUniversalEmbedder` from the given `MPPUniversalEmbedderOptions`.
 *
 * @param options The options to use for configuring the `MPPUniversalEmbedder`.
 * @param error An optional error parameter populated when there is an error in initializing.
 * @return A new instance of `MPPUniversalEmbedder` with the given options. `nil` if there is an
 * error.
 */
- (nullable instancetype)initWithOptions:(MPPUniversalEmbedderOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs embedding extraction on the input text.
 *
 * @param text The string on which embedding extraction is to be performed.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedText:(NSString *)text
                                     error:(NSError **)error NS_SWIFT_NAME(embed(text:));

/**
 * Performs embedding extraction on the input image bytes (e.g. JPEG, PNG contents).
 *
 * @param imageBytes The raw compressed image data.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedImageBytes:(NSData *)imageBytes
                                           error:(NSError **)error
    NS_SWIFT_NAME(embed(imageBytes:));

/**
 * Performs embedding extraction on the input `MPPImage` (e.g. UIImage).
 *
 * @param image The image on which embedding extraction is to be performed.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedImage:(MPPImage *)image
                                      error:(NSError **)error NS_SWIFT_NAME(embed(image:));

/**
 * Performs embedding extraction on the input audio float samples.
 *
 * @param audioData The float buffer containing the audio samples.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedAudioFloatBuffer:(MPPFloatBuffer *)audioData
                                                 error:(NSError **)error
    NS_SWIFT_NAME(embed(audioFloatBuffer:));

/**
 * Performs embedding extraction on the input `MPPAudioData`.
 *
 * @param audioData The audio data object containing the audio samples.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedAudio:(MPPAudioData *)audioData
                                      error:(NSError **)error NS_SWIFT_NAME(embed(audio:));

/**
 * Performs embedding extraction on multi-modal parts of a content block.
 *
 * Each element in the content array must be one of `NSString`, `MPPImage`, `MPPAudioData`,
 * or `MPPTaskPart` (like `MPPTextPart`, `MPPImagePart`, `MPPAudioPart`).
 *
 * @param content An array of content objects or parts.
 * @param error An optional error parameter populated when there is an error.
 * @return A `MPPEmbeddingResult` object that contains the embeddings.
 */
- (nullable MPPEmbeddingResult *)embedContent:(NSArray<id> *)content
                                        error:(NSError **)error NS_SWIFT_NAME(embed(content:));

/**
 * Utility function to compute cosine similarity between two embeddings.
 *
 * @param embedding1 One of the two embeddings between whom cosine similarity is to be computed.
 * @param embedding2 One of the two embeddings between whom cosine similarity is to be computed.
 * @param error An optional error parameter populated when there is an error.
 * @return An `NSNumber` holding the cosine similarity of type `double`.
 */
+ (nullable NSNumber *)cosineSimilarityBetweenEmbedding1:(MPPEmbedding *)embedding1
                                           andEmbedding2:(MPPEmbedding *)embedding2
                                                   error:(NSError **)error
    NS_SWIFT_NAME(cosineSimilarity(embedding1:embedding2:));

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
