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
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioDataFormat.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioRecord.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * A wrapper class for input audio samples used in on-device machine learning using MediaPipe Task
 * library.
 *
 * Internally this class uses a ring buffer to hold input audio data. Clients could feed input audio
 * data via `load` method and access the aggregated audio samples via `buffer` property.
 *
 * Note that this class can only handle input audio in Float.
 * TODO: Investigate if `AVAudioFormatInt32` should be supported. Only necessary if you can change
 * the input format of `AVAudioEngine` or load audio files using `AVAudioFormatInt32`.
 */
NS_SWIFT_NAME(AudioData)
@interface MPPAudioData : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) MPPAudioDataFormat *format;

/**
 * A copy of all the internal buffer elements in order with the most recent elements appearing at
 * the end of the buffer.
 */
@property(nonatomic, readonly) MPPFloatBuffer *buffer;

/** Capacity of the `AudioData` buffer in number of elements. */
@property(nonatomic, readonly) NSUInteger bufferLength;

/**
 * Initializes a new instance of `AudioData` with the given `AudioFormat` and sample count.
 *
 * The `AudioData` stores data in a ring buffer of length sampleCount * MPPAudioFormat.channelCount.
 *
 * @param format An instance of `AudioFormat` that defines the number of channels and sample rate
 * required by the TFLite model.
 * @param sampleCount The number of samples `AudioData` can store at any given time. The
 * `sampleCount` provided will be used to calculate the buffer length of `AudioData` (with
 * `bufferLength = format.channelCount * sampleCount`.
 *
 * @return A new instance of `AudioData` with the given audio format and sample count.
 */
- (instancetype)initWithFormat:(MPPAudioDataFormat *)format
                   sampleCount:(NSUInteger)sampleCount NS_DESIGNATED_INITIALIZER;

/**
 * Loads the internal buffer of `AudioData` with a slice of the audio samples contained in the
 * provided `FloatBuffer`.
 *
 * New data from the input buffer is appended to the end of the buffer by shifting out any old data
 * from the beginning of the buffer if needed to make space. If the length of the new data to be
 * copied is more than the capacity of the buffer, only the most recent data of the `AudioData`'s
 * buffer will be copied from the input buffer.
 *
 * @param sourceBuffer  A float buffer whose values are to be loaded into `AudioData`. For
 * multi-channel input, the array must be interleaved.
 * @param offset Starting index in the source buffer from which elements should be copied.
 * @param length The number of elements to be copied.
 *
 * @return A boolean indicating if the load operation succeeded.
 */
- (BOOL)loadBuffer:(MPPFloatBuffer *)buffer
            offset:(NSUInteger)offset
            length:(NSUInteger)length
             error:(NSError **)error NS_SWIFT_NAME(load(buffer:offset:length:));

/**
 * Loads the internal buffer of `AudioData` with the audio samples currently held by the audio
 * record.
 *
 * New data from the audio record is appended to the end of the buffer by shifting out any old data
 * from the beginning of the buffer if needed to make space. If the length of the new data to be
 * copied is more than the capacity of the buffer, only the most recent data of the `AudioData`'s
 * buffer will be copied from the input buffer.
 *
 * @param audioRecord `MPPAudioRecord` whose values are to be loaded into `AudioData`. For
 * multi-channel input, the audio record must hold interleaved data.
 *
 * @return A boolean indicating if the load operation succeeded.
 */
- (BOOL)loadAudioRecord:(MPPAudioRecord *)audioRecord
                  error:(NSError **)error NS_SWIFT_NAME(load(audioRecord:));

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
