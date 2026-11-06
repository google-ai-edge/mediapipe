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

import AVFoundation
import Foundation

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Continuously records microphone samples into a thread-safe circular buffer.
///
/// - Note: `AudioRecord` is `@unchecked Sendable` because `AVAudioEngine` does not conform to
///   `Sendable`, and mutable ring-buffer state (`floatRingBuffer` and `globalError`) shared between
///   the realtime `AVAudioNodeTapBlock` callback and reader threads is synchronized via
///   `lock.withLock`. `AVAudioEngine` lifecycle methods (`startRecording()`, `stop()`) are kept
///   outside `lock` because `AVAudioNode.removeTap(onBus:)` and `AVAudioEngine.stop()` wait for
///   in-flight tap callbacks that acquire `lock`.
public final class AudioRecord: @unchecked Sendable {
  /// Maximum supported audio channel count (`2`).
  public static let maximumChannelCount = 2

  private let lock = NSLock()
  private let floatRingBuffer: FloatRingBuffer
  private var globalError: Error?
  private let audioEngine = AVAudioEngine()

  /// Audio format specifying the channel count and sample rate of recorded samples.
  public let audioDataFormat: AudioDataFormat

  /// Maximum number of samples retained across all channels.
  public let bufferLength: Int

  /// Creates a microphone recorder backed by a ring buffer of capacity `bufferLength`.
  ///
  /// - Throws: `TasksError` with `.invalidArgument` if `channelCount` is outside `1...2` or
  ///   `bufferLength` is not a positive multiple of `channelCount`.
  public init(audioDataFormat: AudioDataFormat, bufferLength: Int) throws {
    guard (1...Self.maximumChannelCount).contains(audioDataFormat.channelCount) else {
      throw TasksError.invalidArgument(
        "The channel count provided does not match the supported channel count. "
          + "Only channel counts in the range [1 : \(Self.maximumChannelCount)] are supported."
      )
    }
    guard bufferLength > 0, bufferLength.isMultiple(of: audioDataFormat.channelCount) else {
      throw TasksError.invalidArgument(
        "The buffer length provided (\(bufferLength)) is not a positive multiple of channel "
          + "count (\(audioDataFormat.channelCount))."
      )
    }

    self.audioDataFormat = audioDataFormat
    self.bufferLength = bufferLength
    self.floatRingBuffer = FloatRingBuffer(length: bufferLength)
  }

  /// Starts tapping microphone samples using `AVAudioEngine`.
  public func startRecording() throws {
    guard !audioEngine.isRunning else {
      throw TasksError(
        code: .audioRecordNotTappingMic,
        message:
          "Recording of microphone samples is already in progress. You can stop recording "
          + "using `stop()`."
      )
    }

    #if os(iOS)
      switch AVAudioSession.sharedInstance().recordPermission {
      case .denied:
        throw TasksError(
          code: .audioRecordPermissionDenied,
          message: "Record permission was denied by the user."
        )
      case .undetermined:
        throw TasksError(
          code: .audioRecordPermissionUndetermined,
          message:
            "Record permissions are undetermined. Use AVAudioSession's "
            + "requestRecordPermission() to request audio record permission from the user."
        )
      case .granted:
        break
      @unknown default:
        throw TasksError(
          code: .audioRecordPermissionUndetermined,
          message: "Record permission state is unknown."
        )
      }
    #endif

    try startTappingMicrophone()
  }

  /// Stops recording audio and clears buffered samples.
  public func stop() throws {
    stopAndResetAudioEngine()
    lock.withLock {
      floatRingBuffer.clear()
      globalError = nil
    }
    #if os(iOS)
      try AVAudioSession.sharedInstance().setActive(
        false,
        options: .notifyOthersOnDeactivation
      )
    #endif
  }

  /// Reads `length` samples starting at `offset` from the oldest sample.
  public func read(offset: Int, length: Int) throws -> FloatBuffer {
    guard audioEngine.isRunning else {
      throw TasksError(
        code: .audioRecordNotTappingMic,
        message:
          "Recording of microphone samples is not in progress. You can start recording "
          + "microphone samples using `startRecording()`."
      )
    }
    return try readBuffer(offset: offset, length: length)
  }

  /// Thread-safe ring buffer read used by tests and `read(offset:length:)`.
  package func readBuffer(offset: Int, length: Int) throws -> FloatBuffer {
    try lock.withLock {
      if let globalError {
        throw globalError
      }
      return try floatRingBuffer.floatBuffer(offset: offset, length: length)
    }
  }

  /// Thread-safe ring buffer write used by tests.
  package func loadBuffer(_ floatBuffer: FloatBuffer, offset: Int, length: Int) throws {
    try lock.withLock {
      try floatRingBuffer.load(floatBuffer: floatBuffer, offset: offset, length: length)
      globalError = nil
    }
  }

  private func startTappingMicrophone() throws {
    stopAndResetAudioEngine()
    #if os(iOS)
      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord)
      try session.setActive(true, options: .notifyOthersOnDeactivation)
    #endif

    let inputNode = audioEngine.inputNode
    let inputFormat = inputNode.outputFormat(forBus: 0)

    guard
      let recordingFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: audioDataFormat.sampleRate,
        channels: AVAudioChannelCount(audioDataFormat.channelCount),
        interleaved: true
      ),
      let converter = AVAudioConverter(from: inputFormat, to: recordingFormat)
    else {
      throw TasksError.internalError("Failed to configure AVAudioConverter for microphone tap.")
    }

    inputNode.installTap(
      onBus: 0,
      bufferSize: AVAudioFrameCount(bufferLength),
      format: inputFormat
    ) { [weak self] pcmBuffer, _ in
      guard let self else { return }
      do {
        let converted = try Self.convert(pcmBuffer: pcmBuffer, using: converter)
        try self.loadAudioPCMBuffer(converted)
      } catch {
        self.lock.withLock {
          self.globalError = error
        }
      }
    }

    audioEngine.prepare()
    try audioEngine.start()
  }

  private func stopAndResetAudioEngine() {
    audioEngine.inputNode.removeTap(onBus: 0)
    audioEngine.stop()
    audioEngine.reset()
  }

  package func loadAudioPCMBuffer(_ pcmBuffer: AVAudioPCMBuffer) throws {
    let frameLength = Int(pcmBuffer.frameLength)
    guard frameLength > 0 else {
      throw TasksError.invalidArgument("PCM buffer frameLength must be greater than zero.")
    }
    guard
      pcmBuffer.format.commonFormat == .pcmFormatFloat32,
      Int(pcmBuffer.format.channelCount) == audioDataFormat.channelCount,
      pcmBuffer.format.isInterleaved || audioDataFormat.channelCount == 1,
      let channelData = pcmBuffer.floatChannelData
    else {
      throw TasksError.invalidArgument(
        "Invalid PCM buffer format; expected interleaved Float32 matching configured channel "
          + "count."
      )
    }

    let sampleCount = frameLength * audioDataFormat.channelCount
    let sampleBuffer = FloatBuffer(data: channelData[0], length: sampleCount)
    try loadBuffer(sampleBuffer, offset: 0, length: sampleCount)
  }

  package static func convert(
    pcmBuffer: AVAudioPCMBuffer,
    using audioConverter: AVAudioConverter
  ) throws -> AVAudioPCMBuffer {
    let sampleRateRatio =
      audioConverter.outputFormat.sampleRate / audioConverter.inputFormat.sampleRate
    let frameCapacity = AVAudioFrameCount(
      ceil(Double(pcmBuffer.frameLength) * sampleRateRatio)
    )

    guard
      let outPCMBuffer = AVAudioPCMBuffer(
        pcmFormat: audioConverter.outputFormat,
        frameCapacity: max(1, frameCapacity)
      )
    else {
      throw TasksError.internalError("Failed to allocate converted AVAudioPCMBuffer.")
    }

    var conversionError: NSError?
    var hasProvidedInput = false
    let status = audioConverter.convert(to: outPCMBuffer, error: &conversionError) {
      _, outStatus in
      guard !hasProvidedInput else {
        outStatus.pointee = .noDataNow
        return nil
      }
      hasProvidedInput = true
      outStatus.pointee = .haveData
      return pcmBuffer
    }

    if status == .error {
      let description =
        conversionError?.localizedDescription
        ?? "An error occurred while processing incoming audio frames."
      throw TasksError.internalError(description)
    }
    guard outPCMBuffer.frameLength > 0 else {
      throw TasksError.internalError("Audio converter produced zero output frames.")
    }
    return outPCMBuffer
  }
}

extension AudioData {
  /// Loads the latest samples from `audioRecord`.
  public func load(audioRecord: AudioRecord) throws {
    guard audioRecord.audioDataFormat == format else {
      throw TasksError.invalidArgument(
        "The provided audio record has incompatible audio format."
      )
    }
    let recordedBuffer = try audioRecord.read(offset: 0, length: audioRecord.bufferLength)
    try load(buffer: recordedBuffer, offset: 0, length: recordedBuffer.length)
  }
}
