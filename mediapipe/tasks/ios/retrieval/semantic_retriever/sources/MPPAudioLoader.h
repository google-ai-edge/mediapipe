// Copyright 2026 The MediaPipe Authors. All Rights Reserved.
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

NS_ASSUME_NONNULL_BEGIN

/**
 * A utility class to load audio files from file paths on iOS.
 */
NS_SWIFT_NAME(AudioLoader)
@interface MPPAudioLoader : NSObject

/**
 * Loads an audio file from the given file path and converts it to PCM Float32 format.
 */
+ (nullable MPPAudioData *)loadAudioFromFilePath:(NSString *)filePath
                                           error:(NSError **)error
    NS_SWIFT_NAME(loadAudio(fromFilePath:));

/**
 * Loads audio from in-memory data bytes and converts it to PCM Float32 format.
 */
+ (nullable MPPAudioData *)loadAudioFromData:(NSData *)data error:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
