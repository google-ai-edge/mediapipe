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

#import "mediapipe/tasks/ios/audio/audio_embedder/sources/MPPAudioEmbedderResult.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRunningMode.h"
#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

@class MPPAudioEmbedder;

/**
 * This protocol defines an interface for the delegates of `AudioEmbedder` object to receive
 * results of asynchronous embedding extraction on audio stream (i.e, when `runningMode` =
 * `.audioStream`).
 *
 * The delegate of `AudioEmbedder` must adopt `AudioEmbedderStreamDelegate`` protocol.
 * The methods in this protocol are optional.
 */
NS_SWIFT_NAME(AudioEmbedderStreamDelegate)
@protocol MPPAudioEmbedderStreamDelegate <NSObject>

@optional
/**
 * This method notifies a delegate that the results of asynchronous embedding extraction on
 * an audio stream submitted to the `AudioEmbedder` is available.
 *
 * This method is called on a private serial queue created by the `AudioEmbedder`
 * for performing the asynchronous delegates calls.
 *
 * @param audioEmbedder The audio embedder which performed the embedding extraction.
 * This is useful to test equality when there are multiple instances of `AudioEmbedder`.
 * @param result An `AudioEmbedderResult` object that contains a list of generated audio embeddings.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates when the input
 * audio stream was sent to the audio embedder.
 * @param error An optional error parameter populated when there is an error in performing embedding
 * extraction on the input audio stream data.
 */
- (void)audioEmbedder:(MPPAudioEmbedder *)audioEmbedder
    didFinishEmbeddingWithResult:(nullable MPPAudioEmbedderResult *)result
         timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                           error:(nullable NSError *)error
    NS_SWIFT_NAME(audioEmbedder(_:didFinishEmbedding:timestampInMilliseconds:error:));
@end

/**
 * Options for setting up a `AudioEmbedder`.
 */
NS_SWIFT_NAME(AudioEmbedderOptions)
@interface MPPAudioEmbedderOptions : MPPTaskOptions <NSCopying>

/**
 * Running mode of the image embedder task. Defaults to `.audioClips`.
 * `AudioEmbedder` can be created with one of the following running modes:
 *  1. `.audioClips`: The mode for performing embedding extraction on independent audio clips.
 *  2. `.audioStream`: The mode for performing embedding extraction on an audio stream, such as from
 * a microphone.
 */
@property(nonatomic) MPPAudioRunningMode runningMode;

/**
 * An object that confirms to `AudioEmbedderStreamDelegate`` protocol. This object must
 * implement `audioEmbedder(_:didFinishEmbeddingWithResult:timestampInMilliseconds:error:)` to
 * receive the results of asynchronous embedding extraction on an audio stream (i.e, when
 * `runningMode = .audioStream`).
 */
@property(nonatomic, weak, nullable) id<MPPAudioEmbedderStreamDelegate> audioEmbedderStreamDelegate;

/**
 * @brief Sets whether L2 normalization should be performed on the returned embeddings.
 * Use this option only if the model does not already contain a native L2_NORMALIZATION TF Lite Op.
 * In most cases, this is already the case and L2 norm is thus achieved through TF Lite inference.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL l2Normalize;

/**
 * @brief Sets whether the returned embedding should be quantized to bytes via scalar quantization.
 * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is guaranteed to
 * have value in [-1.0, 1.0]. Use the `l2Normalize` property if this is not the case.
 *
 * `NO` by default.
 */
@property(nonatomic) BOOL quantize;

@end

NS_ASSUME_NONNULL_END
