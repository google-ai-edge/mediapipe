// Copyright 2024 The MediaPipe Authors.
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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierResult.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRunningMode.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPAudioClassifier;

/**
 * This protocol defines an interface for the delegates of `AudioClassifier` object to receive
 * results of asynchronous classification of audio stream (i.e, when `runningMode` =
 * `.audioStream`).
 *
 * The delegate of `AudioClassifier` must adopt `AudioClassifierStreamDelegate` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(AudioClassifierStreamDelegate)
@protocol MPPAudioClassifierStreamDelegate <NSObject>

@optional
/**
 * This method notifies a delegate that the results of asynchronous classification of
 * an audio stream submitted to the `AudioClassifier` is available.
 *
 * This method is called on a private serial queue created by the `AudioClassifier`
 * for performing the asynchronous delegates calls.
 *
 * @param AudioClassifier The audio classifier which performed the classification.
 * This is useful to test equality when there are multiple instances of `AudioClassifier`.
 * @param result An `AudioClassifierResult` object that contains a list of audio classifications.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the audio
 * clip was sent to the audio classifier.
 * @param error An optional error parameter populated when there is an error in performing audio
 * classification on the input audio stream data.
 */
- (void)audioClassifier:(MPPAudioClassifier *)AudioClassifier
    didFinishClassificationWithResult:(nullable MPPAudioClassifierResult *)result
              timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                                error:(nullable NSError *)error
    NS_SWIFT_NAME(audioClassifier(_:didFinishClassification:timestampInMilliseconds:error:));
@end

/**
 * Options for setting up a `AudioClassifier`.
 */
NS_SWIFT_NAME(AudioClassifierOptions)
@interface MPPAudioClassifierOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the audio classifier task. Defaults to `.image`.
 * `AudioClassifier` can be created with one of the following running modes:
 *  1. `.audioClips`: The mode for performing classification on independent audio clips.
 *  2. `.audioStream`: The mode for performing classification on an audio stream, such as from a
 * microphone.
 */
@property(nonatomic) MPPAudioRunningMode runningMode;

/**
 * An object that confirms to `AudioClassifierStreamDelegate` protocol. This object must
 * implement `objectDetector(_:didFinishDetectionWithResult:timestampInMilliseconds:error:)` to
 * receive the results of asynchronous classification on audio stream (i.e, when `runningMode =
 * .audioStream`).
 */
@property(nonatomic, weak, nullable) id<MPPAudioClassifierStreamDelegate>
    audioClassifierStreamDelegate;

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
