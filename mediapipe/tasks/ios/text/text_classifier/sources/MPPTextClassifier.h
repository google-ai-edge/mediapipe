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
#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifierOptions.h"
#import "mediapipe/tasks/ios/text/text_classifier/sources/MPPTextClassifierResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs classification on text.
 *
 * This API expects a TFLite model with (optional) [TFLite Model
 * Metadata](https://www.tensorflow.org/lite/convert/metadata")that contains the mandatory
 * (described below) input tensors, output tensor, and the optional (but recommended) label
 * items as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
 *
 * Metadata is required for models with int32 input tensors because it contains the input
 * process unit for the model's Tokenizer. No metadata is required for models with string
 * input tensors.
 *
 * Input tensors
 *  - Three input tensors `kTfLiteInt32` of shape `[batch_size xbert_max_seq_len]`
 *    representing the input ids, mask ids, and segment ids. This input signature requires
 *    a Bert Tokenizer process unit in the model metadata.
 *  - Or one input tensor `kTfLiteInt32` of shape `[batch_size xmax_seq_len]` representing
 *    the input ids. This input signature requires a Regex Tokenizer process unit in the
 *    model metadata.
 *  - Or one input tensor (`kTfLiteString`) that is shapeless or has shape `[1]` containing
 *    the input string.
 *
 * At least one output tensor (`kTfLiteFloat32/kBool`) with:
 *  - `N` classes and shape `[1 x N]`
 *  - optional (but recommended) label map(s) as AssociatedFiles with type TENSOR_AXIS_LABELS,
 *    containing one label per line. The first such AssociatedFile (if any) is used to fill the
 *    `categoryName` field of the results. The `displayName` field is filled from the
 *    AssociatedFile (if any) whose locale matches the `displayNamesLocale` field of the
 *    `MPPTextClassifierOptions` used at creation time ("en" by default, i.e. English). If none of
 *    these are available, only the `index` field of the results will be filled.
 */
NS_SWIFT_NAME(TextClassifier)
@interface MPPTextClassifier : NSObject

/**
 * Creates a new instance of `MPPTextClassifier` from an absolute path to a TensorFlow Lite
 * model file stored locally on the device and the default `MPPTextClassifierOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text classifier.
 *
 * @return A new instance of `MPPTextClassifier` with the given model path. `nil` if there is an
 * error in initializing the text classifier.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPTextClassifier` from the given `MPPTextClassifierOptions`.
 *
 * @param options The options of type `MPPTextClassifierOptions` to use for configuring the
 * `MPPTextClassifier`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text classifier.
 *
 * @return A new instance of `MPPTextClassifier` with the given options. `nil` if there is an
 * error in initializing the text classifier.
 */
- (nullable instancetype)initWithOptions:(MPPTextClassifierOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs classification on the input text.
 *
 * @param text The `NSString` on which classification is to be performed.
 * @param error An optional error parameter populated when there is an error in performing
 * classification on the input text.
 *
 * @return  A `MPPTextClassifierResult` object that contains a list of text classifications.
 */
- (nullable MPPTextClassifierResult *)classifyText:(NSString *)text
                                             error:(NSError **)error NS_SWIFT_NAME(classify(text:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
