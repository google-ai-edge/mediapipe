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
 * Category is a util class that contains a label, its display name, a float value as score, and the
 * index of the label in the corresponding label file. Typically it's used as the result of
 * classification tasks.
 */
NS_SWIFT_NAME(ResultCategory)
@interface MPPCategory : NSObject

/**
 * The index of the label in the corresponding label file. Set to -1 if the index is
 * not set.
 */
@property(nonatomic, readonly) NSInteger index;

/** Confidence score for this class. */
@property(nonatomic, readonly) float score;

/** The label of this category object. */
@property(nonatomic, readonly, nullable) NSString *categoryName;

/**
 * The display name of the label, which may be translated for different locales. For example, a
 * label, "apple", may be translated into Spanish for display purpose, so that the display name is
 * "manzana".
 */
@property(nonatomic, readonly, nullable) NSString *displayName;

/**
 * Initializes a new `ResultCategory` with the given index, score, category name and display name.
 *
 * @param index The index of the label in the corresponding label file.
 * @param score The probability score of this label category.
 * @param categoryName The label of this category object.
 * @param displayName The display name of the label.
 *
 * @return An instance of `ResultCategory` initialized with the given index, score, category name
 * and display name.
 */
- (instancetype)initWithIndex:(NSInteger)index
                        score:(float)score
                 categoryName:(nullable NSString *)categoryName
                  displayName:(nullable NSString *)displayName NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
