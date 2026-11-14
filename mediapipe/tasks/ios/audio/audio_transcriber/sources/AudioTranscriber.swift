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
package import third_party_mediapipe_tasks_c_core_mp_status

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Performs streaming automatic speech recognition (ASR) on mono PCM float audio samples.
///
/// - Important: `AudioTranscriber` maintains stateful streaming buffers and is **not** thread-safe.
///   Synchronize access externally if calling methods from multiple threads or tasks.
public final class AudioTranscriber {
  private var cTranscriber: MpAudioTranscriberPtr?

  /// Creates a streaming transcriber configured with `options`.
  public init(options: AudioTranscriberOptions) throws {
    self.cTranscriber = try options.withCOptions { cOptions in
      var mutableOptions = cOptions
      var transcriberPtr: MpAudioTranscriberPtr?
      try MpStatus.check(defaultMessage: "Failed to create AudioTranscriber.") { errorMessage in
        MpAudioTranscriberCreate(&mutableOptions, &transcriberPtr, &errorMessage)
      }
      guard let transcriberPtr else {
        throw TasksError.internalError("MpAudioTranscriberCreate returned a null handle.")
      }
      return transcriberPtr
    }
  }

  /// Creates a streaming transcriber from a local `.litertlm` or `.task` model asset path.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: AudioTranscriberOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    if let cTranscriber { _ = MpAudioTranscriberClose(cTranscriber, nil) }
  }

  /// Releases the underlying engine; later calls throw `TasksError` with `.failedPrecondition`.
  public func close() throws {
    guard let transcriber = cTranscriber else { return }
    cTranscriber = nil
    try MpStatus.check(defaultMessage: "Failed to close AudioTranscriber.") {
      MpAudioTranscriberClose(transcriber, &$0)
    }
  }

  /// Feeds mono PCM float samples normalized to `[-1.0, 1.0]` at `options.sampleRateHz` into the
  /// active stream, finalizing the stream when `endOfStream` is `true`.
  public func transcribe(
    samples: [Float],
    endOfStream: Bool = false
  ) throws -> AudioTranscriberResult {
    try samples.withUnsafeBufferPointer { buffer in
      try transcribe(samples: buffer, endOfStream: endOfStream)
    }
  }

  /// Feeds a contiguous buffer of mono PCM float samples in `[-1.0, 1.0]` without copying
  /// `samples`, finalizing the stream when `endOfStream` is `true`.
  public func transcribe(
    samples: UnsafeBufferPointer<Float>,
    endOfStream: Bool = false
  ) throws -> AudioTranscriberResult {
    let transcriber = try activeTranscriber()
    var cResult = MpAudioTranscriberResult()
    defer { MpAudioTranscriberCloseResult(&cResult) }
    var emptyPlaceholder: Float = 0
    try withUnsafePointer(to: &emptyPlaceholder) { placeholderPtr in
      let samplesPtr: UnsafePointer<Float>
      if samples.isEmpty {
        samplesPtr = placeholderPtr
      } else {
        guard let baseAddress = samples.baseAddress else {
          throw TasksError.invalidArgument(
            "Audio samples buffer baseAddress must not be nil when count > 0.")
        }
        samplesPtr = baseAddress
      }
      try MpStatus.check(defaultMessage: "Failed to transcribe audio samples.") { errorMessage in
        MpAudioTranscriberTranscribe(
          transcriber, samplesPtr, samples.count, endOfStream, &cResult, &errorMessage)
      }
    }
    return AudioTranscriberResult(cResult: cResult)
  }

  /// Flushes any buffered audio and emits final segments for the current stream.
  public func finishStream() throws -> AudioTranscriberResult {
    let transcriber = try activeTranscriber()
    var cResult = MpAudioTranscriberResult()
    defer { MpAudioTranscriberCloseResult(&cResult) }
    try MpStatus.check(defaultMessage: "Failed to finish audio transcription stream.") {
      MpAudioTranscriberFinishStream(transcriber, &cResult, &$0)
    }
    return AudioTranscriberResult(cResult: cResult)
  }

  /// Preloads the underlying inference engine to reduce first-chunk latency.
  public func prewarm() throws {
    let transcriber = try activeTranscriber()
    try MpStatus.check(defaultMessage: "Failed to prewarm AudioTranscriber.") {
      MpAudioTranscriberPrewarm(transcriber, &$0)
    }
  }

  private func activeTranscriber() throws -> MpAudioTranscriberPtr {
    guard let cTranscriber else {
      throw TasksError.failedPrecondition("AudioTranscriber has been closed.")
    }
    return cTranscriber
  }
}
