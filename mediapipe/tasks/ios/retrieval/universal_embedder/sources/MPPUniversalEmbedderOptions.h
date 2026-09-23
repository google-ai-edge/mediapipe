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

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Activation data type for model execution.
 */
typedef NS_ENUM(NSUInteger, MPPActivationDataType) {
  MPPActivationDataTypeDefault NS_SWIFT_NAME(default) = 0,
      MPPActivationDataTypeFloat32 NS_SWIFT_NAME(float32) = 1,
          MPPActivationDataTypeFloat16 NS_SWIFT_NAME(float16) = 2,
              MPPActivationDataTypeInt16 NS_SWIFT_NAME(int16) = 3, MPPActivationDataTypeInt8 NS_SWIFT_NAME(int8) = 4,
              } NS_SWIFT_NAME(ActivationDataType);

/**
 * Options for setting up a `MPPUniversalEmbedder`.
 */
NS_SWIFT_NAME(UniversalEmbedderOptions)
@interface MPPUniversalEmbedderOptions : MPPTaskOptions <NSCopying>

/**
 * Whether to L2-normalize the output embedding vector.
 *
 * `YES` by default.
 */
@property(nonatomic) BOOL l2Normalize;

/**
 * Optional maximum sequence length (in tokens) for text encoder signatures. If > 0,
 * the engine will automatically select signatures up to this length.
 *
 * Defaults to 0, which means disabled.
 */
@property(nonatomic) NSInteger maxInputLength;

/**
 * Optional vision tokens per image for vision encoder signatures. If > 0,
 * the engine will automatically select the smallest vision encoder signatures.
 *
 * Defaults to 0, which means disabled.
 */
@property(nonatomic) NSInteger visionTokensPerImage;

/**
 * Activation data type for model execution.
 */
@property(nonatomic) MPPActivationDataType activationDataType;

/**
 * Optional directory path for storing compiled model cache artifacts.
 */
@property(nonatomic, copy, nullable) NSString *cacheDir NS_SWIFT_NAME(cacheDirectory);

@end

NS_ASSUME_NONNULL_END
