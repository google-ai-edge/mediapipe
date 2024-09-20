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

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioData.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

typedef NSDictionary<NSNumber *, MPPAudioData *> TimestampedAudioData;

NS_ASSUME_NONNULL_BEGIN

/**
 * Represents the audio data in a stream created by splitting an audio file into chunks. Timestamp
 * indicates the starting timestamp of the audio data in the audio file.
 */
@interface MPPTimestampedAudioData : NSObject

/** Audio data */
@property(nonatomic, readonly) MPPAudioData *audioData;

/** Timestamp (in milliseconds) of the audio data in the audio file. */
@property(nonatomic, readonly) NSInteger timestampInMilliseconds;

/**
 * Creates a new instance of `MPPTimestampedAudioData` from the given audio data and its timestamp
 * (in milliseconds) in an audio file.
 *
 * @param audioData An instance of `MPPAudioData`.
 * @param timestampInMilliseconds The timestamp (in milliseconds) of the audio data in the audio
 * file.
 *
 * @return Newly created time stamped audio data.
 */
- (instancetype)initWithAudioData:(MPPAudioData *)audioData
          timestampInMilliseconds:(NSInteger)timestampInMilliseconds;

@end

/** Utils for operations on `AVAudioFile`. */
@interface AVAudioFile (TestUtils)

/**
 * Creates an array of `MPPTimestampedAudioData` to mimic streaming an audio file by splitting the
 * file so that the number of samples in each audio data in the array can fit to the number of
 * samples the model takes as input. The audio data objects are ordered by monotonically increasing
 * timestamps in the returned array.
 *
 * @param fileInfo `FileInfo` with name and type of the audio file stored in the bundle.
 * @param modelSampleCount  Model's input size in number of samples.
 * @param modelSampleRate Sample rate of the model.
 *
 * @return Newly created array of `MPPTimestampedAudioData`.
 */
+ (NSArray<MPPTimestampedAudioData *> *)
    streamedAudioBlocksFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                            modelSampleCount:(NSInteger)modelSampleCount
                             modelSampleRate:(double)modelSampleRate;

@end

NS_ASSUME_NONNULL_END
