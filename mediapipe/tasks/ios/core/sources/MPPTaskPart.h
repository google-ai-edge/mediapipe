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

NS_ASSUME_NONNULL_BEGIN

@class MPPEmbeddingResult;
@class MPPTaskPart;

/**
 * Protocol for generating embeddings across different modalities using task content parts.
 */
NS_SWIFT_NAME(EmbeddingProvider)
@protocol MPPEmbeddingProvider <NSObject>

/**
 * Generates a high-dimensional vector embedding for the given list of task parts or objects.
 */
- (nullable MPPEmbeddingResult *)embedContent:(NSArray<id> *)content
                                        error:(NSError **)error NS_SWIFT_NAME(embed(content:));

@end

/**
 * Represents a part of a multi-modal content block.
 */
NS_SWIFT_NAME(TaskPart)
@interface MPPTaskPart : NSObject

@end

/**
 * Text part representing the text content block of a multi-modal block.
 */
NS_SWIFT_NAME(TextPart)
@interface MPPTextPart : MPPTaskPart

@property(nonatomic, readonly, copy) NSString *text;

- (instancetype)initWithText:(NSString *)text;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * Image part representing the image content block of a multi-modal block.
 */
NS_SWIFT_NAME(ImagePart)
@interface MPPImagePart : MPPTaskPart

/**
 * The filesystem path or resource identifier handle (e.g., asset URI) for the image.
 * May be provided alongside in-memory `data` to persist a lightweight handle while embedding
 * from memory.
 */
@property(nonatomic, readonly, copy, nullable) NSString *filePath;

/**
 * The raw in-memory image buffer, if initialized with data.
 * Nil if initialized from a file path alone or deserialized from persistent storage.
 */
@property(nonatomic, readonly, copy, nullable) NSData *data;

/**
 * Initializes a new `MPPImagePart` using a path or resource handle.
 *
 * @param filePath The path or resource handle for the image.
 * @return An instance of `MPPImagePart` configured with the file path.
 */
- (instancetype)initWithFilePath:(NSString *)filePath;

/**
 * Initializes a new `MPPImagePart` using in-memory image buffer data.
 *
 * @param data The raw image data bytes.
 * @return An instance of `MPPImagePart` configured with in-memory data.
 */
- (instancetype)initWithData:(NSData *)data;

/**
 * Initializes a new `MPPImagePart` with both a resource handle and transient in-memory image data.
 *
 * @param filePath The optional path or resource identifier handle to persist in storage.
 * @param data The optional raw image data bytes to use for in-memory embedding.
 * @return An instance of `MPPImagePart`.
 */
- (instancetype)initWithFilePath:(nullable NSString *)filePath
                            data:(nullable NSData *)data NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * Audio part representing the audio content block of a multi-modal block.
 */
NS_SWIFT_NAME(AudioPart)
@interface MPPAudioPart : MPPTaskPart

/**
 * The filesystem path or resource identifier handle (e.g., asset URI) for the audio.
 * May be provided alongside in-memory `data` to persist a lightweight handle while embedding
 * from memory.
 */
@property(nonatomic, readonly, copy, nullable) NSString *filePath;

/**
 * The raw in-memory audio buffer, if initialized with data.
 * Nil if initialized from a file path alone or deserialized from persistent storage.
 */
@property(nonatomic, readonly, copy, nullable) NSData *data;

/**
 * Initializes a new `MPPAudioPart` using a path or resource handle.
 *
 * @param filePath The path or resource handle for the audio.
 * @return An instance of `MPPAudioPart` configured with the file path.
 */
- (instancetype)initWithFilePath:(NSString *)filePath;

/**
 * Initializes a new `MPPAudioPart` using in-memory audio buffer data.
 *
 * @param data The raw audio data bytes.
 * @return An instance of `MPPAudioPart` configured with in-memory data.
 */
- (instancetype)initWithData:(NSData *)data;

/**
 * Initializes a new `MPPAudioPart` with both a resource handle and transient in-memory audio data.
 *
 * @param filePath The optional path or resource identifier handle to persist in storage.
 * @param data The optional raw audio data bytes to use for in-memory embedding.
 * @return An instance of `MPPAudioPart`.
 */
- (instancetype)initWithFilePath:(nullable NSString *)filePath
                            data:(nullable NSData *)data NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
