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
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

NS_ASSUME_NONNULL_BEGIN

/** Helper utility for initializing `MPPAudioData` for MediaPipe iOS audio library tests. */
@interface MPPAudioData (TestUtils)

/**
 * Initializes an `MPPAudioData` from channel count, sample rate and sample count.
 *
 * @param channelCount Number of channels.
 * @param sampleRate Sample rate.
 * @param sampleCount Sample count.
 *
 * @return The `MPPAudioData` object with the specified channel count, sample rate and sample count.
 */
- (instancetype)initWithChannelCount:(NSUInteger)channelCount
                          sampleRate:(double)sampleRate
                         sampleCount:(NSUInteger)sampleCount;

/**
 * Creats an `MPPAudioData` and loads it with samples from an audio file specified by `fileInfo`.
 *
 * @param fileInfo `MPPFileInfo` with name and type of the audio file stored in the bundle.
 *
 * @return The `MPPAudioData` object loaded with samples from the input audio file.
 */
- (instancetype)initWithFileInfo:(MPPFileInfo *)fileInfo;
@end

NS_ASSUME_NONNULL_END
