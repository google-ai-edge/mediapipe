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

#import "mediapipe/tasks/ios/text/text_summarizer/sources/MPPTextSummarizerOptions.h"
#import "mediapipe/tasks/ios/text/text_summarizer/sources/MPPTextSummarizerResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Performs summarization on text.
 *
 * This API expects a LiteRT-LM model asset bundle.
 */
NS_SWIFT_NAME(TextSummarizer)
@interface MPPTextSummarizer : NSObject

/**
 * Creates a new instance of `MPPTextSummarizer` from an absolute path to a model
 * asset bundle stored locally on the device and the default `MPPTextSummarizerOptions`.
 *
 * @param modelPath An absolute path to a model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text summarizer.
 *
 * @return A new instance of `MPPTextSummarizer` with the given model path. `nil` if there is an
 * error in initializing the text summarizer.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `MPPTextSummarizer` from the given `MPPTextSummarizerOptions`.
 *
 * @param options The options of type `MPPTextSummarizerOptions` to use for configuring the
 * `MPPTextSummarizer`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * text summarizer.
 *
 * @return A new instance of `MPPTextSummarizer` with the given options. `nil` if there is an
 * error in initializing the text summarizer.
 */
- (nullable instancetype)initWithOptions:(MPPTextSummarizerOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs summarization on the input text.
 *
 * @param text The `NSString` on which summarization is to be performed.
 * @param error An optional error parameter populated when there is an error in performing
 * summarization on the input text.
 *
 * @return  A `MPPTextSummarizerResult` object that contains the complete summary result.
 */
- (nullable MPPTextSummarizerResult *)summarizeText:(NSString *)text
                                              error:(NSError **)error
    NS_SWIFT_NAME(summarize(text:));

/**
 * Performs summarization on the input `text` and returns the result
 * incrementally via the callback.
 *
 * @param text The input text to be summarized.
 * @param completionHandler The completion handler invoked with the incremental summarization result
 * or an error.
 * @param error An optional error parameter populated when there is an error in initiating
 * the summarization on the input text.
 *
 * @return `YES` if the stream summarization was successfully initiated, `NO` otherwise.
 */
- (BOOL)summarizeStreamingWithText:(NSString *)text
                 completionHandler:(void (^)(MPPTextSummarizerStreamResult *_Nullable result,
                                             NSError *_Nullable error))completionHandler
                             error:(NSError **)error
    NS_SWIFT_NAME(summarizeStreaming(text:completion:));

/**
 * Shuts down the `MPPTextSummarizer` when all the work is done.
 *
 * @param error An optional error parameter populated when there is an error in shutting down the
 * text summarizer.
 * @return `YES` if the `MPPTextSummarizer` was successfully closed, `NO` otherwise.
 */
- (BOOL)closeWithError:(NSError **)error NS_SWIFT_NAME(close());

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
