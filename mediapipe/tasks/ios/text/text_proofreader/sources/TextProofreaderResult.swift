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
package import third_party_mediapipe_tasks_c_text_text_proofreader_text_proofreader_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
#endif

/// Edit operation transforming a segment of input text into proofread output.
public enum CorrectionType: Int, Sendable, Equatable, Hashable, CaseIterable {
  /// Segment appears unchanged in both the original and proofread text.
  case same = 0
  /// Segment was inserted into the proofread text.
  case insertion = 1
  /// Segment was removed from the original text.
  case deletion = 2

  package init(cType: MpCorrectionType) {
    switch cType {
    case MP_CORRECTION_TYPE_INSERTION:
      self = .insertion
    case MP_CORRECTION_TYPE_DELETION:
      self = .deletion
    default:
      self = .same
    }
  }
}

/// A single unchanged, inserted, or deleted text segment produced during proofreading.
public struct Correction: Sendable, Equatable, Hashable {
  /// Edit kind applied to `text`.
  public let type: CorrectionType
  /// Text span associated with the edit operation.
  public let text: String

  /// Creates a `Correction` for the given edit operation and text span.
  public init(type: CorrectionType, text: String) {
    self.type = type
    self.text = text
  }

  package init(cCorrection: MpCorrection) {
    self.type = CorrectionType(cType: cCorrection.type)
    self.text = cCorrection.text.map { String(cString: $0) } ?? ""
  }
}

/// Completed output returned by synchronous proofreading.
public struct TextProofreaderResult: Sendable, Equatable, Hashable {
  /// Proofread text with all grammar and spelling edits applied.
  public let proofreadText: String
  /// Ordered token-level diff segments between the original input and `proofreadText`.
  public let corrections: [Correction]

  /// Creates a `TextProofreaderResult` with the generated text and diff segments.
  public init(proofreadText: String, corrections: [Correction]) {
    self.proofreadText = proofreadText
    self.corrections = corrections
  }

  package init(cResult: MpTextProofreaderResult) {
    self.proofreadText = cResult.proofread_text.map { String(cString: $0) } ?? ""
    self.corrections = Array(
      cArray: UnsafePointer(cResult.corrections),
      count: cResult.corrections_count,
      transform: Correction.init(cCorrection:))
  }
}

/// Incremental chunk or terminal diff returned during streaming proofreading.
public struct TextProofreaderStreamResult: Sendable, Equatable, Hashable {
  /// Newly generated text fragment for this streaming update.
  public let chunk: String
  /// Ordered diff segments populated when `isDone` is `true`, or `nil` on intermediate chunks.
  public let corrections: [Correction]?
  /// Indicates whether streaming generation has finished.
  public let isDone: Bool

  /// Creates a `TextProofreaderStreamResult` with a text chunk, optional corrections, and status.
  public init(chunk: String, corrections: [Correction]? = nil, isDone: Bool) {
    self.chunk = chunk
    self.corrections = corrections
    self.isDone = isDone
  }

  package init(cResult: MpTextProofreaderStreamResult) {
    self.chunk = cResult.chunk.map { String(cString: $0) } ?? ""
    self.isDone = cResult.done
    if let correctionsPtr = UnsafePointer(cResult.corrections) {
      self.corrections = Array(
        cArray: correctionsPtr,
        count: cResult.corrections_count,
        transform: Correction.init(cCorrection:))
    } else {
      self.corrections = cResult.done ? [] : nil
    }
  }
}
