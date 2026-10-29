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

#import "mediapipe/tasks/ios/core/sources/MPPTaskOptions.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Model architecture backend family for `MPPAudioTranscriber`.
 */
typedef NS_ENUM(NSUInteger, MPPAudioTranscriberModelArchitecture) {
  /** Automatically infer the model architecture from the model asset. */
  MPPAudioTranscriberModelArchitectureAuto = 0,
  /** Open-weight ASR models executed via the LiteRT Omni ASR runtime. */
  MPPAudioTranscriberModelArchitectureOmni = 1,
} NS_SWIFT_NAME(AudioTranscriberModelArchitecture);

/**
 * Options for configuring a MediaPipe `MPPAudioTranscriber` task.
 */
NS_SWIFT_NAME(AudioTranscriberOptions)
@interface MPPAudioTranscriberOptions : MPPTaskOptions <NSCopying>

/**
 * Input audio sample rate in Hz. If different from 16000 Hz, `MPPAudioTranscriber` automatically
 * resamples incoming audio to 16000 Hz.
 *
 * Default is 16000.0.
 */
@property(nonatomic) double sampleRateHz;

/**
 * Whether to emit interim (non-final) transcription segments during streaming.
 *
 * Default is `NO`.
 */
@property(nonatomic) BOOL enableInterimResults;

/**
 * Model architecture family. Defaults to `MPPAudioTranscriberModelArchitectureAuto`.
 */
@property(nonatomic) MPPAudioTranscriberModelArchitecture modelArchitecture;

/**
 * Optional system/user prompt override for generative ASR models.
 */
@property(nonatomic, copy, nullable) NSString *prompt;

/**
 * Optional directory path for caching compiled model artifacts.
 *
 * To disable caching, set this to `@":nocache"`.
 */
@property(nonatomic, copy, nullable) NSString *cacheDir NS_SWIFT_NAME(cacheDirectory);

@end

NS_ASSUME_NONNULL_END
