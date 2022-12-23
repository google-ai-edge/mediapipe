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

#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifierResult.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"
#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifierOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A Mediapipe iOS Text Classifier.
 */
NS_SWIFT_NAME(TextClassifier)
@interface MPPTextClassifier : NSObject

/**
 * Creates a new instance of `MPPTextClassifier` from an absolute path to a TensorFlow Lite model
 * file stored locally on the device.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @param error An optional error parameter populated when there is an error in initializing
 * the text classifier.
 *
 * @return A new instance of `MPPTextClassifier` with the given model path. `nil` if there is an
 * error in initializing the text classifier.
 */
- (instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPTextClassifier` from the given text classifier options.
 *
 * @param options The options to use for configuring the `MPPTextClassifier`.
 * @param error An optional error parameter populated when there is an error in initializing
 * the text classifier.
 *
 * @return A new instance of `MPPTextClassifier` with the given options. `nil` if there is an error
 * in initializing the text classifier.
 */
- (instancetype)initWithOptions:(MPPTextClassifierOptions *)options error:(NSError **)error;

- (nullable MPPTextClassifierResult *)classifyWithText:(NSString *)text error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
