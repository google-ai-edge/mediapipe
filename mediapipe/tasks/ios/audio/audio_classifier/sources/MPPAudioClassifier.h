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

#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierOptions.h"
#import "mediapipe/tasks/ios/audio/audio_classifier/sources/MPPAudioClassifierResult.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * @brief Class that performs classification on audio data.
 *
 * This API expects a TFLite model with mandatory TFLite Model Metadata that contains the mandatory
 * AudioProperties of the solo input audio tensor and the optional (but recommended) category labels
 * as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
 *
 * Input tensor
 *  (kTfLiteFloat32)
 *  - input audio buffer of size `[batch * samples]`.
 *  - batch inference is not supported (`batch` is required to be 1).
 *  - for multi-channel models, the channels must be interleaved.
 *
 * At least one output tensor with:
 *  (kTfLiteFloat32)
 *  - `[1 x N]` array with `N` represents the number of categories.
 *  - optional (but recommended) category labels as AssociatedFiles with type TENSOR_AXIS_LABELS,
 *    containing one label per line. The first such AssociatedFile (if any) is used to fill the
 *    `category_name` field of the results. The `display_name` field is filled from the
 *    AssociatedFile (if any) whose locale matches the `display_names_locale` field of the
 *    `AudioClassifierOptions` used at creation time ("en" by default, i.e. English). If none of
 *    these are available, only the `index` field of the results will be filled.
 */
NS_SWIFT_NAME(AudioClassifier)
@interface MPPAudioClassifier : NSObject

/**
 * Creates a new instance of `AudioClassifier` from an absolute path to a TensorFlow Lite model file
 * stored locally on the device and the default `AudioClassifierOptions`.
 *
 * @param modelPath An absolute path to a TensorFlow Lite model file stored locally on the device.
 *
 * @return A new instance of `AudioClassifier` with the given model path. This method throws an
 * error if the audio classifier initialization fails.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Creates a new instance of `AudioClassifier` from the given `AudioClassifierOptions`.
 *
 * @param options The options of type `AudioClassifierOptions` to use for configuring the
 * `AudioClassifier`.
 *
 * @return A new instance of `AudioClassifier` with the given options. This method throws an error
 * if the audio classifier initialization fails.
 */
- (nullable instancetype)initWithOptions:(MPPAudioClassifierOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Performs classification on the provided audio clip. Only use this method when the
 * `AudioClassifier` is created with the `.audioClips` mode.
 *
 * The audio clip is represented as a `AudioData` object  The method accepts audio clips with
 * various lengths and audio sample rates. The `AudioData` object must be initialized with an
 * `AudioDataFormat` specifying the sample rate and channel count of the audio clip.
 *
 * The input audio clip may be longer than what the model is able to process in a single inference.
 * When this occurs, the input audio clip is split into multiple chunks starting at different
 * timestamps. For this reason, the `AudioClassifierResult` this function returns consists of an
 * array of `ClassificationResult` objects, each associated with a timestamp corresponding to the
 * start (in milliseconds) of the chunk data that was classified, e.g:
 *
 * ClassificationResult #0 (first chunk of data):
 *  timestamp_ms: 0 (starts at 0ms)
 *  classifications #0 (single head model):
 *   category #0:
 *    category_name: "Speech"
 *    score: 0.6
 *   category #1:
 *    category_name: "Music"
 *    score: 0.2
 * ClassificationResult #1 (second chunk of data):
 *  timestamp_ms: 800 (starts at 800ms)
 *  classifications #0 (single head model):
 *   category #0:
 *    category_name: "Speech"
 *    score: 0.5
 *   category #1:
 *    category_name: "Silence"
 *    score: 0.1
 *
 * @param audioClip The audio clip of type `AudioData` on which audio classification is to be
 * performed.
 *
 * @return An `AudioClassifierResult` object that contains a list of audio classifications. This
 * method throws an error if audio classification on the audio block fails.
 */
- (nullable MPPAudioClassifierResult *)classifyAudioClip:(MPPAudioData *)audioClip
                                                   error:(NSError **)error
    NS_SWIFT_NAME(classify(audioClip:));

/**
 * Sends audio data (a block in a continuous audio stream) to perform audio classification. Only use
 * this method when the `AudioClassifier` is created with the `.audioStream` mode. This method is
 * designed to process auido stream data such as the microphone input.
 *
 * This method will return immediately after the input audio data is accepted. The results are
 * delivered asynchronously via delegation. The object which needs to be continuously notified of
 * the available results of audio classification must confirm to `AudioClassifierLiveStreamDelegate`
 * protocol and implement the
 * `audioClassifier(_:didFinishClassificationWithResult:timestampInMilliseconds:error:)` delegate
 * method.
 *
 * The audio block is represented as a `AudioData` object. The audio data will be resampled,
 * accumulated, and framed to the proper size for the underlying model to consume. The `AudioData`
 * object must be initialized with an `AudioDataFormat` specifying the sample rate and channel count
 * of the audio stream as well as a timestamp (in milliseconds) to indicate the start time of the
 * input audio block. The timestamps must be monotonically increasing. The input audio block may be
 * longer than what the model is able to process in a single inference. When this occurs, the input
 * audio block is split into multiple chunks. For this reason, the callback may be called multiple
 * times (once per chunk) for each call to this function.
 *
 * @param audioBlock A block of audio data of type `AudioData` in a continuous audio stream.
 * @param timestampInMilliseconds The timestamp (in milliseconds) which indicates the start time of
 * the input audio block
 *
 * @return This method returns successfully if the audio block is submitted for classification.
 * Otherwise, throws an error indicating the reason for failure.
 */
- (BOOL)classifyAsyncAudioBlock:(MPPAudioData *)audioBlock
        timestampInMilliseconds:(NSInteger)timestampInMilliseconds
                          error:(NSError **)error
    NS_SWIFT_NAME(classifyAsync(audioBlock:timestampInMilliseconds:));

/**
 * Closes and cleans up the MediaPipe audio classifier.
 *
 * For audio classifiers initialized with `.audioStream` mode, ensure that this method is called
 * after all audio blocks in an audio stream are sent for inference using
 * `classifyAsync(audioBlock:timestampInMilliseconds:)`. Otherwise, the audio classifier will not
 * process the last audio block (of type `AudioData`) in the stream if its `bufferLength` is shorter
 * than the model's input length. Once an audio classifier is closed, you cannot send any inference
 * requests to it. You must create a new instance of `AudioClassifier` to send any pending requests.
 * Ensure that you are ready to dispose off the audio classifier before this method is invoked.
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
                                                error:(NSError **)error
    NS_SWIFT_NAME(createAudioRecord(channelCount:sampleRate:bufferLength:));

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
