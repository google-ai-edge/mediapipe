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

@class MPPInteractiveSegmenter;

/** Options for setting up a `InteractiveSegmenter`. */
NS_SWIFT_NAME(InteractiveSegmenterOptions)
@interface MPPInteractiveSegmenterOptions : MPPTaskOptions <NSCopying>

/**
 * The locale to use for display names specified through the TFLite Model Metadata, if any. Defaults
 * to English.
 */
@property(nonatomic, copy) NSString *displayNamesLocale;

/** Represents whether to output confidence masks. */
@property(nonatomic) BOOL shouldOutputConfidenceMasks;

/** Represents whether to output category mask. */
@property(nonatomic) BOOL shouldOutputCategoryMask;

@end

NS_ASSUME_NONNULL_END
