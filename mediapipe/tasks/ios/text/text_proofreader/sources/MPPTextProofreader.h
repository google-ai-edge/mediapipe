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

#import "mediapipe/tasks/ios/text/text_proofreader/sources/MPPTextProofreaderOptions.h"
#import "mediapipe/tasks/ios/text/text_proofreader/sources/MPPTextProofreaderResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs proofreading on text.
 *
 * This API expects a LiteRT-LM model asset bundle.
 */
NS_SWIFT_NAME(TextProofreader)
@interface MPPTextProofreader : NSObject

/**
 * Creates a new instance of `MPPTextProofreader` from an absolute path to a model
 * asset bundle stored locally on the device and the default `MPPTextProofreaderOptions`.
 *
 * @param modelPath An absolute path to a model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text proofreader.
 *
 * @return A new instance of `MPPTextProofreader` with the given model path. `nil` if there is an
 * error in initializing the text proofreader.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPTextProofreader` from the given `MPPTextProofreaderOptions`.
 *
 * @param options The options of type `MPPTextProofreaderOptions` to use for configuring the
 * `MPPTextProofreader`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text proofreader.
 *
 * @return A new instance of `MPPTextProofreader` with the given options. `nil` if there is an
 * error in initializing the text proofreader.
 */
- (nullable instancetype)initWithOptions:(MPPTextProofreaderOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs proofreading on the input text.
 *
 * @param text The `NSString` on which proofreading is to be performed.
 * @param error An optional error parameter populated when there is an error in performing
 * proofreading on the input text.
 *
 * @return  A `MPPTextProofreaderResult` object that contains the complete proofread result.
 */
- (nullable MPPTextProofreaderResult *)proofreadText:(NSString *)text
                                               error:(NSError **)error
    NS_SWIFT_NAME(proofread(text:));

/**
 * Performs proofreading on the input `text` and returns the result incrementally via the callback.
 *
 * @param text The input text to be proofread.
 * @param completionHandler The completion handler invoked with the incremental proofreading result
 * or an error.
 * @param error An optional error parameter populated when there is an error in initiating
 * the proofreading on the input text.
 *
 * @return `YES` if the stream proofreading was successfully initiated, `NO` otherwise.
 */
- (BOOL)proofreadStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextProofreaderStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error
    NS_SWIFT_NAME(proofreadStreaming(text:completion:)) NS_SWIFT_DISABLE_ASYNC;

/**
 * Shuts down the `MPPTextProofreader` when all the work is done.
 *
 * @param error An optional error parameter populated when there is an error in shutting down the
 * text proofreader.
 * @return `YES` if the `MPPTextProofreader` was successfully closed, `NO` otherwise.
 */
- (BOOL)closeWithError:(NSError **)error NS_SWIFT_NAME(close());

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
