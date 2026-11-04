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
package import third_party_mediapipe_tasks_c_audio_core_common

/// Stores input audio samples in a circular buffer.
public final class AudioData {
  private let ringBuffer: FloatRingBuffer

  public let format: AudioDataFormat

  /// Snapshot of all internal buffer elements in chronological order (most recent at the end).
  public var buffer: FloatBuffer {
    ringBuffer.floatBuffer
  }

  /// Capacity in number of multi-channel sample frames.
  public var bufferLength: Int {
    guard format.channelCount > 0 else { return 0 }
    return ringBuffer.length / format.channelCount
  }

  /// Creates a zero-initialized buffer with capacity for `sampleCount` frames per channel.
  public init(format: AudioDataFormat, sampleCount: Int) {
    self.format = format
    let totalElements = max(0, sampleCount * format.channelCount)
    self.ringBuffer = FloatRingBuffer(length: totalElements)
  }

  /// Loads a slice of interleaved audio samples from `buffer` into the internal ring buffer.
  public func load(buffer: FloatBuffer, offset: Int, length: Int) throws {
    try ringBuffer.load(floatBuffer: buffer, offset: offset, length: length)
  }

  /// Invokes `body` with a C `MpAudioData` descriptor backed by a contiguous snapshot of the
  /// current audio samples.
  package func withCAudioData<Result>(
    _ body: (MpAudioData) throws -> Result
  ) rethrows -> Result {
    let snapshot = buffer
    return try snapshot.withUnsafeMutableBufferPointer { mutableBuffer in
      let cAudioData = MpAudioData(
        num_channels: Int32(clamping: format.channelCount),
        sample_rate: format.sampleRate,
        audio_data: mutableBuffer.baseAddress,
        audio_data_size: mutableBuffer.count
      )
      return try body(cAudioData)
    }
  }
}
