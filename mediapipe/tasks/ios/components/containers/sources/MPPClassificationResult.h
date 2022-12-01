/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/components/containers/sources/MPPCategory.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskResult.h"

NS_ASSUME_NONNULL_BEGIN

/** Encapsulates list of predicted classes (aka labels) for a given image classifier head. */
NS_SWIFT_NAME(Classifications)
@interface MPPClassifications : NSObject

/**
 * The index of the classifier head these classes refer to. This is useful for multi-head
 * models.
 */
@property(nonatomic, readonly) NSInteger headIndex;

/** The name of the classifier head, which is the corresponding tensor metadata
 * name.
 */
@property(nonatomic, readonly) NSString *headName;

/** The array of predicted classes, usually sorted by descending scores (e.g.from high to low
 * probability). */
@property(nonatomic, readonly) NSArray<MPPCategory *> *categories;

/**
 * Initializes a new `MPPClassifications` with the given head index and array of categories.
 * head name is initialized to `nil`.
 *
 * @param headIndex The index of the image classifier head these classes refer to.
 * @param categories An array of `MPPCategory` objects encapsulating a list of
 * predictions usually sorted by descending scores (e.g. from high to low probability).
 *
 * @return An instance of `MPPClassifications` initialized with the given head index and
 * array of categories.
 */
- (instancetype)initWithHeadIndex:(NSInteger)headIndex
                       categories:(NSArray<MPPCategory *> *)categories;

/**
 * Initializes a new `MPPClassifications` with the given head index, head name and array of
 * categories.
 *
 * @param headIndex The index of the classifier head these classes refer to.
 * @param headName The name of the classifier head, which is the corresponding tensor metadata
 * name.
 * @param categories An array of `MPPCategory` objects encapsulating a list of
 * predictions usually sorted by descending scores (e.g. from high to low probability).
 *
 * @return An object of `MPPClassifications` initialized with the given head index, head name and
 * array of categories.
 */
- (instancetype)initWithHeadIndex:(NSInteger)headIndex
                         headName:(nullable NSString *)headName
                       categories:(NSArray<MPPCategory *> *)categories;

@end

/** Encapsulates results of any classification task. */
NS_SWIFT_NAME(ClassificationResult)
@interface MPPClassificationResult : MPPTaskResult

/** Array of MPPClassifications objects containing classifier predictions per image classifier
 * head.
 */
@property(nonatomic, readonly) NSArray<MPPClassifications *> *classifications;

/**
 * Initializes a new `MPPClassificationResult` with the given array of classifications.
 *
 * @param classifications An Aaray of `MPPClassifications` objects containing classifier
 * predictions per classifier head.
 *
 * @return An instance of MPPClassificationResult initialized with the given array of
 * classifications.
 */
- (instancetype)initWithClassifications:(NSArray<MPPClassifications *> *)classifications
                              timeStamp:(long)timeStamp;

@end

NS_ASSUME_NONNULL_END
