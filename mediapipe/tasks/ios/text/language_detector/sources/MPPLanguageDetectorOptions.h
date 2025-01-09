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
 * Options for setting up a `LanguageDetector`.
 */
NS_SWIFT_NAME(LanguageDetectorOptions)
@interface MPPLanguageDetectorOptions : MPPTaskOptions <NSCopying>

/**
 * The locale to use for display names specified through the TFLite Model Metadata, if any. Defaults
 * to English.
 */
@property(nonatomic, copy) NSString *displayNamesLocale;

/**
 * The maximum number of top-scored classification results to return. If < 0, all available results
 * will be returned. If 0, an invalid argument error is returned.
 */
@property(nonatomic) NSInteger maxResults;

/**
 * Score threshold to override the one provided in the model metadata (if any). Results below this
 * value are rejected.
 */
@property(nonatomic) float scoreThreshold;

/**
 * The allowlist of category names. If non-empty, detection results whose category name is not in
 * this set will be filtered out. Duplicate or unknown category names are ignored. Mutually
 * exclusive with categoryDenylist.
 */
@property(nonatomic, copy) NSArray<NSString *> *categoryAllowlist;

/**
 * The denylist of category names. If non-empty, detection results whose category name is in this
 * set will be filtered out. Duplicate or unknown category names are ignored. Mutually exclusive
 * with categoryAllowlist.
 */
@property(nonatomic, copy) NSArray<NSString *> *categoryDenylist;

@end

NS_ASSUME_NONNULL_END
