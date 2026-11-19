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
package import third_party_mediapipe_tasks_c_audio_audio_classifier_audio_classifier_lib
private import third_party_mediapipe_tasks_c_audio_core_common
package import third_party_mediapipe_tasks_c_core_mp_status

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksAudioCore
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
#endif

/// Categorizes standalone audio clips or continuous audio streams using a MediaPipe model.
///
/// - Important: Instances are not thread-safe (`Sendable` is intentionally omitted);
///   serialize `classify`, `classifyAsync`, and `close` calls from a single execution context.
public final class AudioClassifier {
  private let lock = NSLock()
  private let runningMode: AudioRunningMode
  private var cClassifier: MpAudioClassifierPtr?
  /// Receives the audio stream results. The C task keeps an unretained pointer to it until the task
  /// has been closed. Guarded by `lock` and released together with `cClassifier`, so that a closed
  /// classifier no longer retains the result handler.
  private var streamContext: AudioStreamContext?

  /// Creates an instance from `options`.
  public init(options: AudioClassifierOptions) throws {
    let streamContext: AudioStreamContext?
    switch options.runningMode {
    case .audioClips:
      guard options.audioStreamResultHandler == nil else {
        throw TasksError.invalidArgument(
          "Audio stream result handler must not be set in audio clips mode.")
      }
      streamContext = nil
    case .audioStream:
      guard let handler = options.audioStreamResultHandler else {
        throw TasksError.invalidArgument(
          "Audio stream result handler must be set in audio stream mode.")
      }
      streamContext = AudioStreamContext(handler: handler)
    }

    var created: MpAudioClassifierPtr?
    try options.withCOptions(streamContext: streamContext) { cOptions in
      try MpStatus.check { MpAudioClassifierCreate(&cOptions, &created, &$0) }
    }
    guard let created else {
      throw TasksError.internalError("Failed to create MpAudioClassifier instance.")
    }

    self.runningMode = options.runningMode
    self.cClassifier = created
    self.streamContext = streamContext
  }

  /// Creates an `.audioClips` instance from a local model file.
  public convenience init(modelPath: String) throws {
    let opts = AudioClassifierOptions(baseOptions: BaseOptions(modelAssetPath: modelPath))
    try self.init(options: opts)
  }

  deinit {
    let (classifier, streamContext) = takeClassifierToClose()
    if let classifier {
      // The C task calls the stream context until `MpAudioClassifierClose` has returned.
      withExtendedLifetime(streamContext) { _ = MpAudioClassifierClose(classifier, nil) }
    }
  }

  /// Runs synchronous inference on a standalone clip in `.audioClips` mode.
  public func classify(audioClip: AudioData) throws -> AudioClassifierResult {
    try withPreparedCall(.audioClips, audioClip) { classifier, cAudio in
      var cResult = MpAudioClassifierResult()
      try MpStatus.check { MpAudioClassifierClassify(classifier, &cAudio, &cResult, &$0) }
      defer { MpAudioClassifierCloseResult(&cResult) }
      return AudioClassifierResult(cResult: cResult, timestampInMilliseconds: 0)
    }
  }

  /// Enqueues a stream block tagged with monotonically increasing milliseconds.
  public func classifyAsync(audioBlock: AudioData, timestampInMilliseconds: Int) throws {
    // MediaPipe converts milliseconds to microseconds without an overflow check.
    guard (0...Int(Int64.max / 1000)).contains(timestampInMilliseconds) else {
      throw TasksError.invalidArgument("Timestamp \(timestampInMilliseconds) ms is out of range.")
    }
    try withPreparedCall(.audioStream, audioBlock) { classifier, cAudio in
      try MpStatus.check {
        MpAudioClassifierClassifyAsync(classifier, &cAudio, Int64(timestampInMilliseconds), &$0)
      }
    }
  }

  /// Flushes pending stream packets and shuts down the underlying MediaPipe graph. Releases the
  /// audio stream result handler once the results that are still pending have been delivered.
  public func close() throws {
    let (classifier, streamContext) = takeClassifierToClose()
    guard let classifier else { return }
    // The C task calls the stream context until `MpAudioClassifierClose` has returned.
    try withExtendedLifetime(streamContext) {
      try MpStatus.check(defaultMessage: "Failed to close AudioClassifier.") {
        MpAudioClassifierClose(classifier, &$0)
      }
    }
  }

  /// Creates a microphone recorder matching the requested format and buffer capacity.
  public static func createAudioRecord(
    channelCount: Int, sampleRate: Double, bufferLength: Int
  ) throws -> AudioRecord {
    let format = AudioDataFormat(channelCount: channelCount, sampleRate: sampleRate)
    return try AudioRecord(audioDataFormat: format, bufferLength: bufferLength)
  }

  /// Removes the C classifier and its stream context from this instance. The caller closes the
  /// classifier and keeps the context alive until the close call has returned.
  private func takeClassifierToClose() -> (MpAudioClassifierPtr?, AudioStreamContext?) {
    lock.withLock {
      defer {
        cClassifier = nil
        streamContext = nil
      }
      return (cClassifier, streamContext)
    }
  }

  private func withPreparedCall<R>(
    _ expectedMode: AudioRunningMode, _ audioData: AudioData,
    _ body: (MpAudioClassifierPtr, inout MpAudioData) throws -> R
  ) throws -> R {
    guard runningMode == expectedMode else {
      throw TasksError.invalidArgument(
        "Expected \(expectedMode.displayName) mode, got \(runningMode.displayName).")
    }
    guard let classifier = lock.withLock({ cClassifier }) else {
      throw TasksError.invalidArgument("Task runner is currently not running.")
    }
    guard audioData.format.channelCount > 0, audioData.format.sampleRate > 0,
      audioData.bufferLength > 0
    else {
      throw TasksError.invalidArgument("AudioData must be non-empty with valid format.")
    }
    return try audioData.withCAudioData { cAudioData in
      var mutableData = cAudioData
      return try body(classifier, &mutableData)
    }
  }
}

/// Receives the audio stream results of one `AudioClassifier` from the C result callback.
///
/// The classifier sets `userData` as `result_callback_user_data` and `cCallback` as
/// `result_callback` in its C options. The C task does not retain `userData`, so the classifier
/// must keep this object alive until its `MpAudioClassifierClose` call has returned.
/// `MpAudioClassifierClose` waits for pending results and invokes no callback after it returns.
package final class AudioStreamContext: Sendable {
  private let handler: AudioClassifierOptions.AudioStreamResultHandler
  private let queue = DispatchQueue(
    label: "com.google.mediapipe.tasks.audioClassifier.callback.\(UUID().uuidString)"
  )

  package init(handler: @escaping AudioClassifierOptions.AudioStreamResultHandler) {
    self.handler = handler
  }

  /// The value for `result_callback_user_data` in the C options. Does not retain `self`.
  package var userData: UnsafeMutableRawPointer {
    Unmanaged.passUnretained(self).toOpaque()
  }

  /// The value for `result_callback` in the C options. Delivers the result to the context that its
  /// `user_data` argument points to.
  package static let cCallback: MpAudioClassifierResultCallbackFn = {
    userData, status, errorMessage, cResult in
    receive(userData: userData, status: status, errorMessage: errorMessage, result: cResult)
  }

  /// Forwards the arguments of a C result callback to the context that `userData` points to.
  package static func receive(
    userData: UnsafeMutableRawPointer?,
    status: MpStatus,
    errorMessage: UnsafePointer<CChar>?,
    result: UnsafePointer<MpAudioClassifierResult>?
  ) {
    guard let userData else { return }
    Unmanaged<AudioStreamContext>.fromOpaque(userData).takeUnretainedValue()
      .deliver(status, errorMessage, result)
  }

  /// Decodes `cResult` synchronously before the C pointers expire and delivers the outcome on
  /// `queue`.
  private func deliver(
    _ status: MpStatus,
    _ errorMessage: UnsafePointer<CChar>?,
    _ cResult: UnsafePointer<MpAudioClassifierResult>?
  ) {
    let outcome: Result<AudioClassifierResult, any Error>
    if status != kMpOk {
      outcome = .failure(
        TasksError(
          code: TasksErrorCode(cStatus: status),
          message: errorMessage.map { String(cString: $0) }
        )
      )
    } else if let cResult {
      outcome = .success(AudioClassifierResult(cResult: cResult.pointee))
    } else {
      outcome = .failure(
        TasksError.internalError("Audio stream callback received kMpOk with a null result.")
      )
    }
    let handler = self.handler
    queue.async { handler(outcome) }
  }
}
