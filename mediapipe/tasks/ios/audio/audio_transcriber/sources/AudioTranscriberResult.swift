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

/// A single interim or finalized transcript span produced by `AudioTranscriber`.
public struct TranscriptionSegment: Sendable, Equatable, Hashable {
  /// Recognized text for this segment.
  public let transcript: String

  /// Whether this segment is finalized rather than an interim streaming hypothesis.
  public let isFinal: Bool

  /// Start timestamp of the segment within the audio stream, in milliseconds.
  public let fromTimestampInMilliseconds: Int

  /// End timestamp of the segment within the audio stream, in milliseconds.
  public let toTimestampInMilliseconds: Int

  /// Creates a `TranscriptionSegment` with explicit transcript text, finality, and bounds.
  public init(
    transcript: String,
    isFinal: Bool,
    fromTimestampInMilliseconds: Int,
    toTimestampInMilliseconds: Int
  ) {
    self.transcript = transcript
    self.isFinal = isFinal
    self.fromTimestampInMilliseconds = fromTimestampInMilliseconds
    self.toTimestampInMilliseconds = toTimestampInMilliseconds
  }

  /// Copies a C `MpTranscriptionSegment` into an owned Swift value.
  package init(cSegment: MpTranscriptionSegment) {
    self.init(
      transcript: cSegment.transcript.map { String(cString: $0) } ?? "",
      isFinal: cSegment.is_final,
      fromTimestampInMilliseconds: Int(cSegment.from_timestamp_ms),
      toTimestampInMilliseconds: Int(cSegment.to_timestamp_ms)
    )
  }
}

/// Source-compatibility alias matching the Objective-C bridged segment type name.
public typealias AudioTranscriberSegment = TranscriptionSegment

/// Transcription output returned by `AudioTranscriber` for an audio chunk or stream flush.
public struct AudioTranscriberResult: Sendable, Equatable, Hashable {
  /// Segments emitted during this invocation, in stream order.
  public let segments: [TranscriptionSegment]

  /// Stream timestamp associated with this result, in milliseconds.
  public let timestampInMilliseconds: Int

  /// Concatenated text across all non-empty segments, separated by spaces where needed.
  public var text: String {
    var combined = ""
    for segment in segments where !segment.transcript.isEmpty {
      if !combined.isEmpty && !combined.hasSuffix(" ") && !segment.transcript.hasPrefix(" ") {
        combined.append(" ")
      }
      combined.append(segment.transcript)
    }
    return combined
  }

  /// Creates an `AudioTranscriberResult` from segments and an explicit result timestamp.
  public init(segments: [TranscriptionSegment], timestampInMilliseconds: Int = 0) {
    self.segments = segments
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  /// Copies a C `MpAudioTranscriberResult` into an owned Swift value.
  package init(cResult: MpAudioTranscriberResult) {
    let segmentCount = Int(cResult.segments_count)
    let copiedSegments: [TranscriptionSegment]
    if segmentCount > 0, let cSegments = cResult.segments {
      copiedSegments = (0..<segmentCount).map { index in
        TranscriptionSegment(cSegment: cSegments[index])
      }
    } else {
      copiedSegments = []
    }
    self.init(
      segments: copiedSegments,
      timestampInMilliseconds: copiedSegments.last?.toTimestampInMilliseconds ?? 0
    )
  }
}
