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
package import third_party_mediapipe_tasks_c_text_text_summarizer_text_summarizer_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Complete summary output produced by `TextSummarizer`.
public struct TextSummarizerResult: Sendable, Equatable, Hashable {
  /// Generated summary text.
  public let summary: String

  /// Creates a summarization result with the generated text.
  public init(summary: String) {
    self.summary = summary
  }

  /// Creates a `TextSummarizerResult` copied from a C `MpTextSummarizerResult` struct.
  package init(cResult: MpTextSummarizerResult) {
    self.summary = cResult.summary.map { String(cString: $0) } ?? ""
  }
}

/// Incremental chunk produced during streaming text summarization.
public struct TextSummarizerStreamResult: Sendable, Equatable, Hashable {
  /// Decoded text segment emitted for this streaming step.
  public let chunk: String

  /// Indicates whether streaming summarization has finished.
  public let isDone: Bool

  /// Creates a streaming summarization chunk with its completion state.
  public init(chunk: String, isDone: Bool) {
    self.chunk = chunk
    self.isDone = isDone
  }

  /// Creates a `TextSummarizerStreamResult` copied from a C `MpTextSummarizerStreamResult` struct.
  package init(cResult: MpTextSummarizerStreamResult) {
    self.chunk = cResult.chunk.map { String(cString: $0) } ?? ""
    self.isDone = cResult.done
  }
}
