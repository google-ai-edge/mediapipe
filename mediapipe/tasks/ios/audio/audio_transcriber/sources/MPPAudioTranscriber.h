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

#import "mediapipe/tasks/ios/audio/audio_transcriber/sources/MPPAudioTranscriberOptions.h"
#import "mediapipe/tasks/ios/audio/audio_transcriber/sources/MPPAudioTranscriberResult.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Performs automatic speech recognition (ASR) on streaming mono PCM float audio samples.
 */
NS_SWIFT_NAME(AudioTranscriber)
@interface MPPAudioTranscriber : NSObject

/**
 * Creates a new instance of `MPPAudioTranscriber` from the given `MPPAudioTranscriberOptions`.
 *
 * @param options The options of type `MPPAudioTranscriberOptions` to use for configuring the
 * `MPPAudioTranscriber`.
 * @param error An optional error parameter populated when there is an error in initializing the
 * audio transcriber.
 * @return A new instance of `MPPAudioTranscriber` with the given options. `nil` if there is an
 * error in initializing the audio transcriber.
 */
- (nullable instancetype)initWithOptions:(MPPAudioTranscriberOptions *)options
                                   error:(NSError **)error NS_DESIGNATED_INITIALIZER;

/**
 * Creates a new instance of `MPPAudioTranscriber` from an absolute path to a model asset bundle
 * stored locally on the device and the default `MPPAudioTranscriberOptions`.
 *
 * @param modelPath An absolute path to a model file stored locally on the device.
 * @param error An optional error parameter populated when there is an error in initializing the
 * audio transcriber.
 * @return A new instance of `MPPAudioTranscriber` with the given model path. `nil` if there is an
 * error in initializing the audio transcriber.
 */
- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error;

/**
 * Transcribes a contiguous buffer of mono PCM float audio samples in `[-1.0, 1.0]`.
 *
 * @param samples Pointer to a contiguous array of mono PCM float audio samples.
 * @param length Number of float samples in `samples`.
 * @param endOfStream Whether this chunk marks the end of the audio stream.
 * @param error An optional error parameter populated when there is an error during transcription.
 * @return An `MPPAudioTranscriberResult` containing any emitted segments, or `nil` on error.
 */
- (nullable MPPAudioTranscriberResult *)transcribeFloatBuffer:(nullable const float *)samples
                                                       length:(NSUInteger)length
                                                  endOfStream:(BOOL)endOfStream
                                                        error:(NSError **)error
    NS_SWIFT_NAME(transcribe(samples:count:endOfStream:));

/**
 * Transcribes an array of mono PCM float audio samples in `[-1.0, 1.0]`.
 *
 * @param samples Array of `NSNumber` float values representing mono PCM audio samples.
 * @param endOfStream Whether this chunk marks the end of the audio stream.
 * @param error An optional error parameter populated when there is an error during transcription.
 * @return An `MPPAudioTranscriberResult` containing any emitted segments, or `nil` on error.
 */
- (nullable MPPAudioTranscriberResult *)transcribeSamples:(NSArray<NSNumber *> *)samples
                                              endOfStream:(BOOL)endOfStream
                                                    error:(NSError **)error
    NS_SWIFT_NAME(transcribe(samples:endOfStream:));

/**
 * Flushes any remaining buffered audio and finalizes the current stream.
 *
 * @param error An optional error parameter populated when there is an error finalizing the stream.
 * @return An `MPPAudioTranscriberResult` containing any final emitted segments, or `nil` on error.
 */
- (nullable MPPAudioTranscriberResult *)finishStreamWithError:(NSError **)error
    NS_SWIFT_NAME(finishStream());

/**
 * Prewarms the underlying inference engine to reduce first-chunk latency.
 *
 * @param error An optional error parameter populated when prewarming fails.
 * @return `YES` if prewarming succeeded, `NO` otherwise.
 */
- (BOOL)prewarmWithError:(NSError **)error NS_SWIFT_NAME(prewarm());

/**
 * Releases underlying inference engine and processor resources.
 *
 * @param error An optional error parameter populated when closing fails.
 * @return `YES` if the `MPPAudioTranscriber` was successfully closed, `NO` otherwise.
 */
- (BOOL)closeWithError:(NSError **)error NS_SWIFT_NAME(close());

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
