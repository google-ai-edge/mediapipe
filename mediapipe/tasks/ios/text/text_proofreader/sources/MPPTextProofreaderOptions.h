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

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Options for configuring a MediaPipe text proofreader task.
 */
NS_SWIFT_NAME(TextProofreaderOptions)
@interface MPPTextProofreaderOptions : MPPTaskOptions <NSCopying>

/**
 * The maximum number of tokens for proofread tasks. If set, the result will be truncated if the
 * input and output exceed this value.
 *
 * If not set, then the default maxNumTokens is based on the model's capacity.
 *
 * Default is 0.
 */
@property(nonatomic, assign) NSInteger maxNumTokens NS_SWIFT_NAME(maxTokens);

/**
 * Cache directory for the model weight cache and program cache. By default, it will be set by
 * LiteRT-LM engine based on the model file path.
 *
 * To disable caching, set this to @":nocache".
 */
@property(nonatomic, copy, nullable) NSString *cacheDir NS_SWIFT_NAME(cacheDirectory);

@end

NS_ASSUME_NONNULL_END
