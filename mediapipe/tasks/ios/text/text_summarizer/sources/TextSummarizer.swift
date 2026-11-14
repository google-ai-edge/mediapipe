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
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_text_text_summarizer_text_summarizer_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
#endif

/// Summarizes text using a LiteRT-LM model asset bundle.
///
/// - Note: This class is not thread-safe, and only one summarization request may execute at a time
///   per instance; overlapping calls or calling `close()` while a stream is active throw
///   `TasksError` with `.failedPrecondition`.
public final class TextSummarizer {
  private let lock = NSLock()
  private var cSummarizer: MpTextSummarizerPtr?
  private var isStreaming = false

  /// Creates a text summarizer and initializes the underlying LiteRT-LM engine.
  ///
  /// - Throws: `TasksError` if `options` are invalid or engine initialization fails.
  public init(options: TextSummarizerOptions) throws {
    self.cSummarizer = try options.withCOptions { cOptions in
      var mutableOptions = cOptions
      var summarizerPtr: MpTextSummarizerPtr?
      try MpStatus.check(defaultMessage: "Failed to create TextSummarizer.") { errorMessage in
        MpTextSummarizerCreate(&mutableOptions, &summarizerPtr, &errorMessage)
      }
      guard let summarizerPtr else {
        throw TasksError.internalError("MpTextSummarizerCreate returned a null pointer.")
      }
      return summarizerPtr
    }
  }

  /// Creates a text summarizer from a local `.litertlm` model file path using default options.
  ///
  /// - Throws: `TasksError` if the model cannot be loaded or initialized.
  public convenience init(modelPath: String) throws {
    let baseOptions = BaseOptions(modelAssetPath: modelPath)
    try self.init(options: TextSummarizerOptions(baseOptions: baseOptions))
  }

  deinit {
    if let ptr = cSummarizer {
      _ = MpTextSummarizerClose(ptr, nil)
    }
  }

  /// Releases the underlying engine; subsequent calls are no-ops, and later summarization calls
  /// throw `TasksError` with `.failedPrecondition`.
  ///
  /// - Throws: `TasksError` with `.failedPrecondition` if a streaming request is still in progress.
  public func close() throws {
    let ptr: MpTextSummarizerPtr? = try lock.withLock {
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "Cannot close TextSummarizer while a streaming request is in progress."
        )
      }
      defer { cSummarizer = nil }
      return cSummarizer
    }
    guard let ptr else {
      return
    }
    try MpStatus.check(defaultMessage: "Failed to close TextSummarizer.") {
      MpTextSummarizerClose(ptr, &$0)
    }
  }

  /// Summarizes non-empty `text` synchronously and returns the complete summary.
  ///
  /// - Throws: `TasksError` if `text` is empty, another request is active, or generation fails.
  public func summarize(text: String) throws -> TextSummarizerResult {
    let summarizerPtr = try activeSummarizer()
    var cResult = MpTextSummarizerResult()
    defer { MpTextSummarizerCloseResult(&cResult) }
    try text.withCString { utf8Text in
      try MpStatus.check(defaultMessage: "Failed to summarize text.") { errorMessage in
        MpTextSummarizerSummarize(summarizerPtr, utf8Text, &cResult, &errorMessage)
      }
    }
    return TextSummarizerResult(cResult: cResult)
  }

  /// Summarizes non-empty `text` incrementally, yielding chunks until completion (`isDone`).
  ///
  /// Cancelling the returned `AsyncThrowingStream` stops yielding elements immediately while the
  /// underlying C engine finishes draining the active generation, during which overlapping
  /// summarization calls or `close()` throw `TasksError` with `.failedPrecondition`.
  public func summarizeStreaming(
    text: String
  ) -> AsyncThrowingStream<TextSummarizerStreamResult, any Error> {
    AsyncThrowingStream { continuation in
      let summarizerPtr: MpTextSummarizerPtr
      do {
        summarizerPtr = try beginStreaming()
      } catch {
        continuation.finish(throwing: error)
        return
      }
      let context = StreamContext(owner: self, continuation: continuation)
      continuation.onTermination = { _ in context.detachContinuation() }
      let userData = context.retainForCallback()
      do {
        try text.withCString { utf8Text in
          try MpStatus.check(defaultMessage: "Failed to start streaming summarization.") {
            errorMessage in
            MpTextSummarizerSummarizeStreaming(
              summarizerPtr,
              utf8Text,
              textSummarizerStreamCallback,
              userData,
              &errorMessage
            )
          }
        }
      } catch {
        context.finish(error: error)
      }
    }
  }

  private func activeSummarizer() throws -> MpTextSummarizerPtr {
    try lock.withLock {
      guard let ptr = cSummarizer else {
        throw TasksError(code: .failedPrecondition, message: "TextSummarizer is already closed.")
      }
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "A streaming summarization request is still in progress."
        )
      }
      return ptr
    }
  }

  private func beginStreaming() throws -> MpTextSummarizerPtr {
    try lock.withLock {
      guard let ptr = cSummarizer else {
        throw TasksError(code: .failedPrecondition, message: "TextSummarizer is already closed.")
      }
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "A streaming summarization request is still in progress."
        )
      }
      isStreaming = true
      return ptr
    }
  }

  fileprivate func markStreamFinished() {
    lock.withLock { isStreaming = false }
  }
}

private final class StreamContext: @unchecked Sendable {
  private let lock = NSLock()
  private var owner: TextSummarizer?
  private var continuation: AsyncThrowingStream<TextSummarizerStreamResult, any Error>.Continuation?
  private var retainedSelf: Unmanaged<StreamContext>?

  init(
    owner: TextSummarizer,
    continuation: AsyncThrowingStream<TextSummarizerStreamResult, any Error>.Continuation
  ) {
    self.owner = owner
    self.continuation = continuation
  }

  func retainForCallback() -> UnsafeMutableRawPointer {
    lock.withLock {
      let unmanaged = Unmanaged.passRetained(self)
      retainedSelf = unmanaged
      return unmanaged.toOpaque()
    }
  }

  func detachContinuation() {
    lock.withLock {
      continuation = nil
    }
  }

  func yield(_ result: TextSummarizerStreamResult) {
    let activeContinuation = lock.withLock { continuation }
    activeContinuation?.yield(result)
  }

  func finish(finalResult: TextSummarizerStreamResult? = nil, error: (any Error)? = nil) {
    let (activeContinuation, ownerRelease, toRelease) = lock.withLock {
      let activeContinuation = continuation
      let ownerRelease = owner.map(Unmanaged.passRetained)
      let toRelease = retainedSelf
      continuation = nil
      retainedSelf = nil
      owner = nil
      return (activeContinuation, ownerRelease, toRelease)
    }

    ownerRelease?.takeUnretainedValue().markStreamFinished()
    if let finalResult { activeContinuation?.yield(finalResult) }
    activeContinuation?.finish(throwing: error)
    // Final summarizer release joins the engine callback pool, so it must not run on that pool.
    DispatchQueue.global().async { ownerRelease?.release() }
    toRelease?.release()
  }
}

private func textSummarizerStreamCallback(
  userData: UnsafeMutableRawPointer?,
  cResult: UnsafePointer<MpTextSummarizerStreamResult>?,
  errorMessage: UnsafePointer<CChar>?
) {
  guard let userData else { return }
  let context = Unmanaged<StreamContext>.fromOpaque(userData).takeUnretainedValue()
  withExtendedLifetime(context) {
    if let errorMessage {
      context.finish(error: TasksError.internalError(String(cString: errorMessage)))
      return
    }
    guard let cResult else {
      context.finish(error: TasksError.internalError("Streaming callback received null result."))
      return
    }
    let streamResult = TextSummarizerStreamResult(cResult: cResult.pointee)
    if streamResult.isDone {
      context.finish(finalResult: streamResult)
    } else {
      context.yield(streamResult)
    }
  }
}
