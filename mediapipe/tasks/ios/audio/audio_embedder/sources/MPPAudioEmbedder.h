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

#import "mediapipe/tasks/ios/audio/audio_embedder/sources/MPPAudioEmbedderOptions.h"
#import "mediapipe/tasks/ios/audio/audio_embedder/sources/MPPAudioEmbedderResult.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs audio embedding extraction on audio clips or audio stream.
 *
 * This API expects a TFLite model with mandatory TFLite Model Metadata that contains the
 * mandatory AudioProperties of the solo input audio tensor and the optional (but recommended) label
 * items as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
 *
 * Input tensor
 *  (kTfLiteFloat32)
 *  - input audio buffer of size `[batch * samples]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - for multi-channel models, the channels need be interleaved.
 *
 * At least one output tensor with:
 *  (kTfLiteFloat32)
 *  - `N` components corresponding to the `N` dimensions of the returned feature vector for this
 *    output layer
 *  - Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
 */
NS_SWIFT_NAME(AudioEmbedder)
@interface MPPAudioEmbedder : NSObject

/**
 * Creates a new instance of `AudioEmbedder` from an absolute path to a TensorFlow Lite model file
 * stored locally on the device and the default `AudioEmbedderOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `AudioEmbedder` with the given model path. This method throws an
 * error if the audio embedder initialization fails.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `AudioEmbedder` from the given `AudioEmbedderOptions`.
 *
 * @param options The options of type `AudioEmbedderOptions` to use for configuring the
 * `AudioEmbedder`.
 *
 * @return A new instance of `AudioEmbedder` with the given options. This method throws an error if
 * the audio embedder initialization fails.
 */
- (nullable instancetype)initWithOptions:(MPPAudioEmbedderOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs embedding extraction on the provided audio clip. Only use this method when the
 * `AudioEmbedder` is created with the `.audioClips` mode.
 *
 * The audio clip is represented as a `AudioData` object  The method accepts audio clips with
 * various lengths and audio sample rates. The `AudioData` object must be initialized with an
 * `AudioDataFormat` specifying the sample rate and channel count of the audio clip.
 *
 * The input audio clip may be longer than what the model is able to process in a single inference.
 * When this occurs, the input audio clip is split into multiple chunks starting at different
 * timestamps. For this reason, the `AudioEmbedderResult` this function returns consists of an
 * array of `EmbeddingResult` objects, each associated with a timestamp corresponding to the
 * start (in milliseconds) of the chunk data on which embedding extraction was performed.
 *
 * @param audioClip An audio clip of type `AudioData` on which embedding extraction is to be
 * performed.
 *
 * @return An `AudioEmbedderResult` object that contains a list of embedding extractions each
 * associated with a timestamp corresponding to the start (in milliseconds) of the chunk data on
 * which embedding extraction was performed. This method throws an error if embedding extraction on
 * the audio clip fails.
 */
- (nullable MPPAudioEmbedderResult *)embedAudioClip:(MPPAudioData *)audioClip
                                              error:(NSError **)error
    NS_SWIFT_NAME(embed(audioClip:));

/**
 * Sends audio data (a block in a continuous audio stream) to the `AudioEmbedder` for embedding
 * extraction. Only use this method when the `AudioEmbedder` is created with the `.audioStream`
 * mode. This method is designed to process auido stream data such as the microphone input.
 *
 * This method will return immediately after the input audio data is accepted. The results are
 * delivered asynchronously via delegation. The object which needs to be continuously notified of
 * the available results of embedding extraction must confirm to `AudioEmbedderLiveStreamDelegate`
 * protocol and implement the
 * `audioEmbedder(_:didFinishEmbeddingWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * The audio block is represented as an `AudioData` object. The audio data will be resampled,
 * accumulated, and framed to the proper size for the underlying model to consume. The `AudioData`
 * object must be initialized with an `AudioDataFormat` specifying the sample rate and channel count
 * of the audio stream as well as a timestamp (in milliseconds) to indicate the start time of the
 * input audio block. The timestamps must be monotonically increasing. The input audio block may be
 * longer than what the model is able to process in a single inference. When this occurs, the input
 * audio block is split into multiple chunks. For this reason, the callback may be called multiple
 * times (once per chunk) for each call to this function.
 *
 * @param audioBlock A block of audio data of type `AudioData` in a continuous audio stream on which
 * embedding extraction is to be performed.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates the start time of
 * the input audio block
 *
 * @return This method returns successfully if the audio block is submitted for embedding
 * extraction. Otherwise, throws an error indicating the reason for failure.
 */
- (BOOL)embedAsyncAudioBlock:(MPPAudioData *)audioBlock
     timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                       error:(NSError **)error
    NS_SWIFT_NAME(embedAsync(audioBlock:timestampInMilliseconds:));

/**
 * Closes and cleans up the MediaPipe audio embedder.
 *
 * For audio embedders initialized with `.audioStream` mode, ensure that this method is called
 * after all audio blocks in an audio stream are sent for inference using
 * `embedAsync(audioBlock:timestampInMilliseconds:)`. Otherwise, the audio embedder will not
 * process the last audio block (of type `AudioData`) in the stream if its `bufferLength` is shorter
 * than the model's input length. Once an audio embedder is closed, you cannot send any inference
 * requests to it. You must create a new instance of `AudioEmbedder` to send any pending requests.
 * Ensure that you are ready to dispose off the audio embedder before this method is invoked.
 *
 * @return Returns successfully if the task was closed. Otherwise, throws an error
 * indicating the reason for failure.
 */
- (BOOL)closeWithError:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates an `AudioRecord` instance to get samples from the audio stream produced by the
 * microphone.
 *
 * The client must call appropriate methods from the audio record to start receiving samples from
 * the microphone.
 *
 * Note that MediaPipe Audio tasks will up/down sample automatically to fit the sample rate required
 * by the model. The default sample rate of the MediaPipe pretrained audio model, Yamnet is 16kHz.
 *
 * @param channelCount Number of channels expected by the client.
 * @param sampleRate Sample rate of the audio expected by the client.
 * @param bufferLength Maximum number of elements the internal buffer of `AudioRecord` can hold at
 * any given point of time. The buffer length must be a multiple of `format.channelCount`.
 *
 * @return An new instance of `AudioRecord` with the given audio format and buffer length. This
 * method throws an error if there is an error in initializing the `AudioRecord`.
 */
+ (MPPAudioRecord *)createAudioRecordWithChannelCount:(NSUInteger)channelCount
                                           sampleRate:(double)sampleRate
                                         bufferLength:(NSUInteger)bufferLength
                                                error:(NSError **)error;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
