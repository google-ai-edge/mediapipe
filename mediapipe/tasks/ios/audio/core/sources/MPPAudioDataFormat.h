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

NS_ASSUME_NONNULL_BEGIN

/**
 * Wraps properties describing the format of the incoming audio samples, namely number of channels
 * and the sample rate.
 */
NS_SWIFT_NAME(AudioDataFormat)

@interface MPPAudioDataFormat : NSObject

/** Number of channels */
@property(nonatomic, readonly) NSUInteger channelCount;

/** Sample rate */
@property(nonatomic, readonly) double sampleRate;

/**
 * Initializes a new `AudioDataFormat` with the given channel count and sample rate.
 *
 * @param channelCount Number of channels.
 * @param sampleRate Sample rate.
 *
 * @return A new instance of `AudioDataFormat` with the given channel count and sample rate.
 */
- (instancetype)initWithChannelCount:(NSUInteger)channelCount sampleRate:(double)sampleRate;

/**
 * Initializes a new `AudioDataFormat` with the default channel count of 1 and the given sample
 * rate.
 *
 * @param sampleRate Sample rate.
 * @return A new instance of `AudioDataFormat` with the default channel count of 1 and the given
 * sample rate.
 */
- (instancetype)initWithSampleRate:(double)sampleRate;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
