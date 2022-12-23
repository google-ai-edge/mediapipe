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

#import "mediapipe/tasks/ios/components/processors/sources/MPPClassifierOptions.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Options to configure MPPTextClassifierOptions.
 */
NS_SWIFT_NAME(TextClassifierOptions)
@interface MPPTextClassifierOptions : MPPTaskOptions

/**
 * Options controlling the behavior of the embedding model specified in the
 * base options.
 */
@property(nonatomic, copy) MPPClassifierOptions *classifierOptions;

// /**
//  * Initializes a new `MPPTextClassifierOptions` with the absolute path to the model file
//  * stored locally on the device, set to the given the model path.
//  *
//  * @discussion The external model file must be a single standalone TFLite file. It could be packed
//  * with TFLite Model Metadata[1] and associated files if they exist. Failure to provide the
//  * necessary metadata and associated files might result in errors. Check the [documentation]
//  * (https://www.tensorflow.org/lite/convert/metadata) for each task about the specific requirement.
//  *
//  * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
//  *
//  * @return An instance of `MPPTextClassifierOptions` initialized to the given model path.
//  */
// - (instancetype)initWithModelPath:(NSString *)modelPath;

@end

NS_ASSUME_NONNULL_END
