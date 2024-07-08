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
#import "mediapipe/tasks/ios/audio/core/sources/MPPAudioDataFormat.h"
#import "mediapipe/tasks/ios/audio/core/sources/MPPFloatBuffer.h"
#import "mediapipe/tasks/ios/test/utils/sources/MPPFileInfo.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Utils for operations on `AVAudioPCMBuffer` needed for tests.
 */
@interface AVAudioPCMBuffer (TestUtils)

@property(nonatomic, readonly, nullable) MPPFloatBuffer *floatBuffer;

/**
 * Creates an `AVAudioPCMBuffer` from the file specified by `fileInfo`. If the format of the audio
 * file is equal to the processing format, the samples loaded from the file are directly returned.
 * If not, the pcm buffer is converted to the processing format.
 *
 * @param fileInfo `FileInfo` with name and type of the audio file stored in the bundle.
 * @param processingFormat The `AVAudioFormat` to which the audio samples are to be converted.
 *
 * @return Newly created audio pcm buffer satisfying the processing format. `nil` if there was an
 * error during conversion.
 */
+ (nullable AVAudioPCMBuffer *)bufferFromAudioFileWithInfo:(MPPFileInfo *)fileInfo
                                          processingFormat:(AVAudioFormat *)processingFormat;

/**
 * Converts the current pcm buffer to a pcm buffer using the specified audio converter
 *
 * @param audioConverter The audio converter to use for conversion.
 *
 * @return Newly created audio pcm buffer satisfying the processing format. `nil` if there was an
 * error during conversion.
 */
- (nullable AVAudioPCMBuffer *)bufferUsingAudioConverter:(AVAudioConverter *)audioConverter;

@end

NS_ASSUME_NONNULL_END
