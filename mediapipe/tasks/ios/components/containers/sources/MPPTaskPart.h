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

#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPTaskPart;

/**
 * Protocol for generating embeddings across different modalities using task content parts.
 */
NS_SWIFT_NAME(EmbeddingProvider)
@protocol MPPEmbeddingProvider <NSObject>

/**
 * Generates a high-dimensional vector embedding for the given list of task parts.
 */
- (nullable NSArray<NSNumber *> *)embedContent:(NSArray<MPPTaskPart *> *)content
                                         error:(NSError **)error;

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

@property(nonatomic, readonly, copy) NSData *imageBytes;

- (instancetype)initWithImageBytes:(NSData *)imageBytes;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/**
 * Audio part representing the audio content block of a multi-modal block.
 */
NS_SWIFT_NAME(AudioPart)
@interface MPPAudioPart : MPPTaskPart

@property(nonatomic, readonly) MPPFloatBuffer *audioData;

- (instancetype)initWithAudioData:(MPPFloatBuffer *)audioData;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
