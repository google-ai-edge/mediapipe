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
  @_exported import MediaPipeTasksCore
#endif

/// Corrects spelling and grammatical errors in input text and computes token-level diffs.
///
/// - Note: `TextProofreader` is not thread-safe. Only one proofreading request may execute at a
///   time per instance; overlapping calls or calling `close()` while a stream is active throw
///   `TasksError` with `.failedPrecondition`.
public final class TextProofreader {
  private let lock = NSLock()
  private var cProofreader: MpTextProofreaderPtr?
  private var isStreaming = false

  /// Creates a `TextProofreader` configured with `options`.
  public init(options: TextProofreaderOptions) throws {
    var proofreaderPtr: MpTextProofreaderPtr?
    try options.withCOptions { cOptions in
      var mutableOptions = cOptions
      try MpStatus.check(defaultMessage: "Failed to create TextProofreader.") { errorMsg in
        MpTextProofreaderCreate(&mutableOptions, &proofreaderPtr, &errorMsg)
      }
    }
    guard let proofreaderPtr else {
      throw TasksError.internalError("MpTextProofreaderCreate succeeded without returning a task.")
    }
    self.cProofreader = proofreaderPtr
  }

  /// Creates a `TextProofreader` with default options using the model file at `modelPath`.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: TextProofreaderOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    if let ptr = cProofreader {
      _ = MpTextProofreaderClose(ptr, nil)
    }
  }

  /// Releases the underlying engine; subsequent calls are no-ops, and later proofreading calls
  /// throw `TasksError` with `.failedPrecondition`.
  ///
  /// - Throws: `TasksError` with `.failedPrecondition` if a streaming request is still in progress.
  public func close() throws {
    let ptr: MpTextProofreaderPtr? = try lock.withLock {
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "Cannot close TextProofreader while a streaming request is in progress."
        )
      }
      defer { cProofreader = nil }
      return cProofreader
    }
    guard let ptr else {
      return
    }
    try MpStatus.check(defaultMessage: "Failed to close TextProofreader.") {
      MpTextProofreaderClose(ptr, &$0)
    }
  }

  /// Proofreads `text` synchronously and returns the corrected text and edit segments.
  public func proofread(text: String) throws -> TextProofreaderResult {
    let proofreaderPtr = try activeProofreader()
    var cResultPtr: UnsafeMutablePointer<MpTextProofreaderResult>?
    defer { MpTextProofreaderCloseResult(cResultPtr) }
    try text.withCString { textPtr in
      try MpStatus.check(defaultMessage: "Failed to proofread text.") { errorMsg in
        MpTextProofreaderProofread(proofreaderPtr, textPtr, &cResultPtr, &errorMsg)
      }
    }
    guard let cResultPtr else {
      throw TasksError.internalError("MpTextProofreaderProofread returned a null result.")
    }
    return TextProofreaderResult(cResult: cResultPtr.pointee)
  }

  /// Streams incremental proofread chunks for `text`, concluding with a final element where
  /// `isDone` is `true` and `corrections` is populated.
  ///
  /// Cancelling the returned `AsyncThrowingStream` stops yielding elements immediately while the
  /// underlying C engine finishes draining the active generation, during which overlapping
  /// proofreading calls or `close()` throw `TasksError` with `.failedPrecondition`.
  public func proofreadStreaming(
    text: String
  ) -> AsyncThrowingStream<TextProofreaderStreamResult, any Error> {
    AsyncThrowingStream { continuation in
      let proofreaderPtr: MpTextProofreaderPtr
      do {
        proofreaderPtr = try beginStreaming()
      } catch {
        continuation.finish(throwing: error)
        return
      }
      let context = StreamContext(owner: self, continuation: continuation)
      continuation.onTermination = { _ in context.clearContinuation() }
      let userData = context.retainForCallback()
      do {
        try text.withCString { textPtr in
          try MpStatus.check(defaultMessage: "Failed to start streaming proofreading.") { err in
            MpTextProofreaderProofreadStreaming(
              proofreaderPtr, textPtr, proofreadStreamCallback, userData, &err)
          }
        }
      } catch {
        context.finish(error: error)
      }
    }
  }

  private func activeProofreader() throws -> MpTextProofreaderPtr {
    try lock.withLock {
      guard let ptr = cProofreader else {
        throw TasksError(code: .failedPrecondition, message: "TextProofreader is already closed.")
      }
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "A streaming proofreading request is still in progress."
        )
      }
      return ptr
    }
  }

  private func beginStreaming() throws -> MpTextProofreaderPtr {
    try lock.withLock {
      guard let ptr = cProofreader else {
        throw TasksError(code: .failedPrecondition, message: "TextProofreader is already closed.")
      }
      guard !isStreaming else {
        throw TasksError(
          code: .failedPrecondition,
          message: "A streaming proofreading request is still in progress."
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
  private var continuation:
    AsyncThrowingStream<TextProofreaderStreamResult, any Error>.Continuation?
  private var owner: TextProofreader?
  private var retainedSelf: Unmanaged<StreamContext>?

  init(
    owner: TextProofreader,
    continuation: AsyncThrowingStream<TextProofreaderStreamResult, any Error>.Continuation
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

  func clearContinuation() {
    lock.withLock {
      continuation = nil
    }
  }

  func yield(_ result: TextProofreaderStreamResult) {
    let activeContinuation = lock.withLock { continuation }
    activeContinuation?.yield(result)
  }

  func finish(finalResult: TextProofreaderStreamResult? = nil, error: (any Error)? = nil) {
    let (activeContinuation, ownerRelease, toRelease) = lock.withLock {
      let activeContinuation = continuation
      let ownerRelease = owner.map(Unmanaged.passRetained)
      let toRelease = retainedSelf
      continuation = nil
      owner = nil
      retainedSelf = nil
      return (activeContinuation, ownerRelease, toRelease)
    }

    ownerRelease?.takeUnretainedValue().markStreamFinished()
    if let finalResult { activeContinuation?.yield(finalResult) }
    activeContinuation?.finish(throwing: error)
    // Releases the owner off the LiteRT-LM callback thread, where `deinit` would deadlock.
    DispatchQueue.global().async { ownerRelease?.release() }
    toRelease?.release()
  }
}

private func proofreadStreamCallback(
  rawContext: UnsafeMutableRawPointer?,
  cResult: UnsafePointer<MpTextProofreaderStreamResult>?,
  errorMessage: UnsafePointer<CChar>?
) {
  guard let rawContext else { return }
  let context = Unmanaged<StreamContext>.fromOpaque(rawContext).takeUnretainedValue()
  withExtendedLifetime(context) {
    if let errorMessage {
      let message = String(cString: errorMessage)
      context.finish(
        error: TasksError.internalError(
          message.isEmpty ? "Streaming proofreading failed." : message))
      return
    }
    guard let cResult else {
      context.finish(error: TasksError.internalError("Streaming callback received null result."))
      return
    }
    let result = TextProofreaderStreamResult(cResult: cResult.pointee)
    if result.isDone {
      context.finish(finalResult: result)
    } else {
      context.yield(result)
    }
  }
}
