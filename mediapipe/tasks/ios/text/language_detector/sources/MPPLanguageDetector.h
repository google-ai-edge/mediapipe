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
#import "mediapipe/tasks/ios/text/language_detector/sources/MPPLanguageDetectorOptions.h"
#import "mediapipe/tasks/ios/text/language_detector/sources/MPPLanguageDetectorResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Predicts the language of an input text.
 *
 * This API expects a TFLite model with [TFLite Model
 * Metadata](https://www.tensorflow.org/lite/convert/metadata")that contains the mandatory
 * (described below) input tensor, output tensor, and the language codes in an AssociatedFile.
 *
 * Metadata is required for models with int32 input tensors because it contains the input
 * process unit for the model's Tokenizer. No metadata is required for models with string
 * input tensors.
 *
 * Input tensor
 *  - One input tensor (`kTfLiteString`) of shape `[1]` containing the input string.
 *
 * Output tensor
 *  - One output tensor (`kTfLiteFloat32`) of shape `[1 x N]` where `N` is the number of languages.
 */
NS_SWIFT_NAME(LanguageDetector)
@interface MPPLanguageDetector : NSObject

/**
 * Creates a new instance of `LanguageDetector` from an absolute path to a TensorFlow Lite
 * model file stored locally on the device and the default `LanguageDetectorOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * language detector.
 *
 * @return A new instance of `LanguageDetector` with the given model path. `nil` if there is an
 * error in initializing the language detector.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `LanguageDetector` from the given `LanguageDetectorOptions`.
 *
 * @param options The options of type `LanguageDetectorOptions` to use for configuring the
 * `LanguageDetector`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * language detector.
 *
 * @return A new instance of `LanguageDetector` with the given options. `nil` if there is an
 * error in initializing the language detector.
 */
- (nullable instancetype)initWithOptions:(MPPLanguageDetectorOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Predicts the language of the input text.
 *
 * @param text The `NSString` for which language is to be predicted.
 * @param error An optional error parameter populated when there is an error in performing
 * language prediction on the input text.
 *
 * @return  A `LanguageDetectorResult` object that contains a list of language predictions.
 */
- (nullable MPPLanguageDetectorResult *)detectText:(NSString *)text
                                             error:(NSError **)error NS_SWIFT_NAME(detect(text:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
