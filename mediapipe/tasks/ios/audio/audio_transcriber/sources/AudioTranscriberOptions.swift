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

import Foundation
package import third_party_mediapipe_tasks_c_audio_audio_transcriber_audio_transcriber_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Model architecture family supported by `AudioTranscriber`.
public enum AudioTranscriberModelArchitecture: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Infers the model architecture automatically from the model asset.
  case auto = 0

  /// Executes open-weight ASR models using the LiteRT Omni ASR runtime.
  case omni = 1

  /// Corresponding C model architecture enum value.
  package var cModelArchitecture: MpAudioTranscriberModelArchitecture {
    switch self {
    case .auto:
      return MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_AUTO
    case .omni:
      return MP_AUDIO_TRANSCRIBER_MODEL_ARCHITECTURE_OMNI
    }
  }
}

/// Configuration options for initializing a MediaPipe `AudioTranscriber` task.
public struct AudioTranscriberOptions: Sendable, Equatable, Hashable {
  /// Shared options for loading the model asset and configuring the hardware delegate.
  public var baseOptions: BaseOptions

  /// Input mono audio sample rate in Hz. Audio at other rates is resampled to 16,000 Hz.
  public var sampleRateHz: Double

  /// Whether streaming transcription emits interim (non-final) segments.
  public var enableInterimResults: Bool

  /// Backend model architecture family used by the transcriber.
  public var modelArchitecture: AudioTranscriberModelArchitecture

  /// Optional prompt override for generative ASR models.
  public var prompt: String?

  /// Optional directory path for caching compiled model artifacts (`":nocache"` disables caching).
  public var cacheDirectory: String?

  /// Creates `AudioTranscriberOptions` with default or custom transcription settings.
  public init(
    baseOptions: BaseOptions = BaseOptions(),
    sampleRateHz: Double = 16_000,
    enableInterimResults: Bool = false,
    modelArchitecture: AudioTranscriberModelArchitecture = .auto,
    prompt: String? = nil,
    cacheDirectory: String? = nil
  ) {
    self.baseOptions = baseOptions
    self.sampleRateHz = sampleRateHz
    self.enableInterimResults = enableInterimResults
    self.modelArchitecture = modelArchitecture
    self.prompt = prompt
    self.cacheDirectory = cacheDirectory
  }

  /// Invokes `body` with a transient C `MpAudioTranscriberOptions` struct whose string and buffer
  /// pointers remain valid for the duration of the closure call.
  package func withCOptions<Result>(
    _ body: (MpAudioTranscriberOptions) throws -> Result
  ) throws -> Result {
    guard sampleRateHz > 0, let cSampleRate = Int32(exactly: sampleRateHz) else {
      throw TasksError.invalidArgument(
        "sampleRateHz must be a positive integer within Int32 range, got \(sampleRateHz)."
      )
    }
    let optionalPrompt = (prompt?.isEmpty == false) ? prompt : nil
    let optionalCacheDirectory = (cacheDirectory?.isEmpty == false) ? cacheDirectory : nil
    return try baseOptions.withCOptions { cBaseOptions in
      try optionalPrompt.withCString { promptPtr in
        try optionalCacheDirectory.withCString { cacheDirPtr in
          var cOptions = MpAudioTranscriberOptions()
          cOptions.base_options = cBaseOptions
          cOptions.sample_rate_hz = cSampleRate
          cOptions.enable_interim_results = enableInterimResults
          cOptions.model_architecture = modelArchitecture.cModelArchitecture
          cOptions.prompt = promptPtr
          cOptions.cache_dir = cacheDirPtr
          return try body(cOptions)
        }
      }
    }
  }
}
