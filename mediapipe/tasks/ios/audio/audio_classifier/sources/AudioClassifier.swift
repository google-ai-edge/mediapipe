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
private import third_party_mediapipe_tasks_c_audio_audio_classifier_audio_classifier_lib
private import third_party_mediapipe_tasks_c_audio_core_common
private import third_party_mediapipe_tasks_c_core_mp_status

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
  private var callbackSlot: Int?

  /// Creates an instance from `options`.
  public init(options: AudioClassifierOptions) throws {
    var allocation: (slot: Int, callback: MpAudioClassifierResultCallbackFn)?
    switch options.runningMode {
    case .audioClips:
      guard options.audioStreamResultHandler == nil else {
        throw TasksError.invalidArgument(
          "Audio stream result handler must not be set in audio clips mode.")
      }
    case .audioStream:
      guard let handler = options.audioStreamResultHandler else {
        throw TasksError.invalidArgument(
          "Audio stream result handler must be set in audio stream mode.")
      }
      allocation = try StreamCallbackPool.acquire(StreamSlot(handler: handler))
    }

    var slotToRelease = allocation?.slot
    defer { if let slot = slotToRelease { StreamCallbackPool.release(slot: slot) } }

    var created: MpAudioClassifierPtr?
    try options.withCOptions(cCallback: allocation?.callback) { cOptions in
      try MpStatus.check { MpAudioClassifierCreate(&cOptions, &created, &$0) }
    }
    guard let created else {
      throw TasksError.internalError("Failed to create MpAudioClassifier instance.")
    }

    slotToRelease = nil
    self.runningMode = options.runningMode
    self.cClassifier = created
    self.callbackSlot = allocation?.slot
  }

  /// Creates an `.audioClips` instance from a local model file.
  public convenience init(modelPath: String) throws {
    let opts = AudioClassifierOptions(baseOptions: BaseOptions(modelAssetPath: modelPath))
    try self.init(options: opts)
  }

  deinit {
    if let classifier = takeClassifierToClose() {
      defer { unregisterCallbackSlot() }
      _ = MpAudioClassifierClose(classifier, nil)
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

  /// Flushes pending stream packets and shuts down the underlying MediaPipe graph.
  public func close() throws {
    guard let classifier = takeClassifierToClose() else { return }
    defer { unregisterCallbackSlot() }
    try MpStatus.check(defaultMessage: "Failed to close AudioClassifier.") {
      MpAudioClassifierClose(classifier, &$0)
    }
  }

  /// Creates a microphone recorder matching the requested format and buffer capacity.
  public static func createAudioRecord(
    channelCount: Int, sampleRate: Double, bufferLength: Int
  ) throws -> AudioRecord {
    let format = AudioDataFormat(sampleRate: sampleRate, channelCount: channelCount)
    return try AudioRecord(audioDataFormat: format, bufferLength: bufferLength)
  }

  private func takeClassifierToClose() -> MpAudioClassifierPtr? {
    lock.withLock {
      defer { cClassifier = nil }
      return cClassifier
    }
  }

  private func unregisterCallbackSlot() {
    let slot: Int? = lock.withLock {
      defer { callbackSlot = nil }
      return callbackSlot
    }
    if let slot { StreamCallbackPool.release(slot: slot) }
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

private final class StreamSlot: Sendable {
  let handler: AudioClassifierOptions.AudioStreamResultHandler
  let queue = DispatchQueue(label: "com.google.mediapipe.tasks.audioClassifier.callback")
  init(handler: @escaping AudioClassifierOptions.AudioStreamResultHandler) {
    self.handler = handler
  }
}

// Maps active `AudioClassifier` instances to static C function-pointer trampolines because
// `MpAudioClassifierResultCallbackFn` is a plain C callback without a `void* user_data` context.
private enum StreamCallbackPool {
  private static let lock = NSLock()
  private nonisolated(unsafe) static var slots = [UnsafeMutableRawPointer?](
    repeating: nil, count: trampolines.count)
  private static let trampolines: [MpAudioClassifierResultCallbackFn] = [
    { run(0, $0, $1) }, { run(1, $0, $1) }, { run(2, $0, $1) }, { run(3, $0, $1) },
    { run(4, $0, $1) }, { run(5, $0, $1) }, { run(6, $0, $1) }, { run(7, $0, $1) },
  ]

  static func acquire(_ entry: StreamSlot) throws -> (Int, MpAudioClassifierResultCallbackFn) {
    try lock.withLock {
      guard let idx = slots.firstIndex(where: { $0 == nil }) else {
        throw TasksError(code: .resourceExhausted, message: "No free audio stream callback slots.")
      }
      slots[idx] = Unmanaged.passRetained(entry).toOpaque()
      return (idx, trampolines[idx])
    }
  }

  static func release(slot: Int) {
    let opaque: UnsafeMutableRawPointer? = lock.withLock {
      defer { slots[slot] = nil }
      return slots[slot]
    }
    if let opaque { Unmanaged<StreamSlot>.fromOpaque(opaque).release() }
  }

  private static func run(
    _ slot: Int, _ status: MpStatus, _ cResult: UnsafeMutablePointer<MpAudioClassifierResult>?
  ) {
    guard
      let entry: StreamSlot = lock.withLock({
        slots[slot].map { Unmanaged<StreamSlot>.fromOpaque($0).takeUnretainedValue() }
      })
    else { return }
    let handler = entry.handler
    let outcome: Result<AudioClassifierResult, any Error>
    if status == kMpOk, let cResult {
      outcome = .success(AudioClassifierResult(cResult: cResult.pointee))
    } else {
      outcome = .failure(TasksError(code: TasksErrorCode(cStatus: status)))
    }
    entry.queue.async { handler(outcome) }
  }
}
