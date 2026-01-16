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

NS_ASSUME_NONNULL_BEGIN

/**
 * Options for setting up a `MPPTextEmbedder`.
 */
NS_SWIFT_NAME(TextEmbedderOptions)
@interface MPPTextEmbedderOptions : MPPTaskOptions <NSCopying>

/**
 * @brief Sets whether L2 normalization should be performed on the returned embeddings.
 * Use this option only if the model does not already contain a native L2_NORMALIZATION TF Lite Op.
 * In most cases, this is already the case and L2 norm is thus achieved through TF Lite inference.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL l2Normalize;

/**
 * @brief Sets whether the returned embedding should be quantized to bytes via scalar quantization.
 * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is guaranteed to
 * have value in [-1.0, 1.0]. Use the `l2Normalize` property if this is not the case.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL quantize;

@end

NS_ASSUME_NONNULL_END
