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

import CoreGraphics
import Foundation
import UIKit
package import third_party_mediapipe_tasks_c_components_containers_rect
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_core_image_processing_options

#if !MEDIAPIPE_SINGLE_MODULE
  import MediaPipeTasksCore
#endif

/// Shared running-mode, orientation, region-of-interest, and C image-preparation helper for
/// MediaPipe Vision Tasks.
package struct VisionTaskRunner: Sendable {
  private enum OrientationRotationDegrees {
    static let up = 0
    static let right = -270
    static let down = -180
    static let left = -90
  }

  package let runningMode: VisionRunningMode
  package let roiAllowed: Bool
  package let normRectAllowed: Bool

  /// Validates `baseOptions` model presence and live-stream callback consistency for `runningMode`.
  package init(
    baseOptions: BaseOptions,
    runningMode: VisionRunningMode,
    roiAllowed: Bool = false,
    normRectAllowed: Bool = true,
    hasLiveStreamHandler: Bool = false
  ) throws {
    guard !baseOptions.modelAssetPath.isEmpty || !(baseOptions.modelAssetBuffer?.isEmpty ?? true)
    else {
      throw TasksError.invalidArgument(
        "BaseOptions must specify a non-empty modelAssetPath or modelAssetBuffer."
      )
    }
    guard (runningMode == .liveStream) == hasLiveStreamHandler else {
      let message =
        runningMode == .liveStream
        ? "The vision task is in live stream mode; a live stream result handler must be set."
        : "The vision task is in image or video mode; a live stream result handler must not be set."
      throw TasksError.invalidArgument(message)
    }
    self.runningMode = runningMode
    self.roiAllowed = roiAllowed
    self.normRectAllowed = normRectAllowed
  }

  /// Validates that `runningMode` matches `expectedMode`.
  package func validate(runningMode expectedMode: VisionRunningMode) throws {
    guard runningMode == expectedMode else {
      throw TasksError.invalidArgument(
        "The vision task is not initialized with \(expectedMode.displayName.lowercased()) mode. "
          + "Current Running Mode: \(runningMode.displayName)"
      )
    }
  }

  /// Converts `timestampInMilliseconds` to the C API timestamp.
  ///
  /// MediaPipe converts milliseconds to microseconds without an overflow check, so values outside
  /// `0...Int64.max / 1000` are rejected.
  package func cTimestamp(_ timestampInMilliseconds: Int) throws -> Int64 {
    guard (0...Int(Int64.max / 1000)).contains(timestampInMilliseconds) else {
      throw TasksError.invalidArgument("Timestamp \(timestampInMilliseconds) ms is out of range.")
    }
    return Int64(timestampInMilliseconds)
  }

  /// Validates `options` and composes `orientation` with `options` into `MpImageProcessingOptions`.
  package func resolveImageProcessingOptions(
    _ options: ImageProcessingOptions? = nil,
    for orientation: UIImage.Orientation
  ) throws -> MpImageProcessingOptions {
    try options?.validate()
    let extraRotation = options?.rotationDegrees ?? 0
    if !normRectAllowed, orientation != .up || extraRotation != 0 {
      throw TasksError.invalidArgument(
        "Unsupported UIImage.Orientation or rotationDegrees; this task only supports .up with 0°."
      )
    }
    if !normRectAllowed || !roiAllowed, options?.regionOfInterest != nil {
      throw TasksError.invalidArgument("This task doesn't support region-of-interest.")
    }

    let orientationDegrees: Int
    switch orientation {
    case .up: orientationDegrees = OrientationRotationDegrees.up
    case .right: orientationDegrees = OrientationRotationDegrees.right
    case .down: orientationDegrees = OrientationRotationDegrees.down
    case .left: orientationDegrees = OrientationRotationDegrees.left
    case .upMirrored, .downMirrored, .leftMirrored, .rightMirrored:
      throw TasksError.invalidArgument(
        "Mirrored UIImage.Orientation (\(orientation.rawValue)) is not supported."
      )
    @unknown default:
      throw TasksError.invalidArgument(
        "Unsupported UIImage.Orientation (\(orientation.rawValue))."
      )
    }

    let (combinedDegrees, overflow) = orientationDegrees.addingReportingOverflow(extraRotation)
    guard !overflow, let cRotation = Int32(exactly: combinedDegrees) else {
      throw TasksError.invalidArgument("Combined rotationDegrees exceeds Int32 range.")
    }

    var cOptions = MpImageProcessingOptions()
    cOptions.rotation_degrees = cRotation
    if let roi = options?.regionOfInterest {
      cOptions.has_region_of_interest = 1
      cOptions.region_of_interest = MpRectF(
        left: Float(roi.origin.x),
        top: Float(roi.origin.y),
        bottom: Float(roi.origin.y + roi.size.height),
        right: Float(roi.origin.x + roi.size.width)
      )
    }
    return cOptions
  }

  /// Validates `expectedMode`, prepares `MpImagePtr` and `MpImageProcessingOptions`, and invokes
  /// `body` within their scoped lifetimes.
  package func withPreparedInput<Result>(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil,
    expectedMode: VisionRunningMode,
    _ body: (MpImagePtr, UnsafePointer<MpImageProcessingOptions>) throws -> Result
  ) throws -> Result {
    try validate(runningMode: expectedMode)
    let cOptions = try resolveImageProcessingOptions(
      imageProcessingOptions, for: image.orientation
    )
    return try image.withCImage { cImage in
      try withUnsafePointer(to: cOptions) { cOptionsPtr in
        try body(cImage, cOptionsPtr)
      }
    }
  }
}

/// Scoped slot in the C live-stream callback pool (`@unchecked Sendable`: `lock` guards state).
///
/// Decodes `CResult` synchronously before C pointers expire, dispatches delivery on a serial queue,
/// and releases the slot via `unregister()` after `Mp*Close` in `deinit`.
///
/// Why this uses a 32-slot static `@convention(c)` trampoline table and `unsafeBitCast` to
/// `CCallback`:
/// 1. MediaPipe's Vision C callbacks (`Mp*ResultCallback` in `tasks/c/vision/`) omit a
///    `void* user_data` context pointer (the same C-API constraint behind `VisionCallbackPool` in
///    Objective-C++ `MPPVisionTaskRunner.h`).
/// 2. Swift `@convention(c)` function pointers cannot capture state or generic type parameters.
/// 3. Every Vision `Mp*ResultCallback` shares the identical 4-argument C ABI layout
///    `(MpStatus, const Mp*Result*, MpImagePtr, int64_t)`, allowing one shared pool to serve all
///    live-stream Vision tasks instead of duplicating 32 trampolines in every task file.
package final class VisionLiveStreamRegistration: @unchecked Sendable {
  private typealias RawCCallback =
    @convention(c) (MpStatus, UnsafeRawPointer?, MpImagePtr?, Int64) -> Void
  private typealias SlotCallback =
    @Sendable (MpStatus, UnsafeRawPointer?, MpImagePtr?, Int64) -> Void

  /// `@unchecked Sendable` is safe because `slots` is exclusively accessed under `lock`.
  private final class PoolStorage: @unchecked Sendable {
    static let shared = PoolStorage()
    private let lock = NSLock()
    private var slots = [SlotCallback?](repeating: nil, count: trampolines.count)

    func allocate(_ callback: @escaping SlotCallback) throws -> Int {
      try lock.withLock {
        guard let index = slots.firstIndex(where: { $0 == nil }) else {
          throw TasksError(
            code: .resourceExhausted,
            message: "Maximum concurrent Vision live-stream tasks (\(slots.count)) exceeded."
          )
        }
        slots[index] = callback
        return index
      }
    }

    func free(_ index: Int) {
      lock.withLock { slots[index] = nil }
    }

    func callback(at index: Int) -> SlotCallback? {
      lock.withLock { slots[index] }
    }
  }

  private static func dispatch(
    _ slot: Int, _ status: MpStatus, _ raw: UnsafeRawPointer?, _ img: MpImagePtr?, _ ts: Int64
  ) {
    PoolStorage.shared.callback(at: slot)?(status, raw, img, ts)
  }

  private static let trampolines: [RawCCallback] = [
    { dispatch(0, $0, $1, $2, $3) }, { dispatch(1, $0, $1, $2, $3) },
    { dispatch(2, $0, $1, $2, $3) }, { dispatch(3, $0, $1, $2, $3) },
    { dispatch(4, $0, $1, $2, $3) }, { dispatch(5, $0, $1, $2, $3) },
    { dispatch(6, $0, $1, $2, $3) }, { dispatch(7, $0, $1, $2, $3) },
    { dispatch(8, $0, $1, $2, $3) }, { dispatch(9, $0, $1, $2, $3) },
    { dispatch(10, $0, $1, $2, $3) }, { dispatch(11, $0, $1, $2, $3) },
    { dispatch(12, $0, $1, $2, $3) }, { dispatch(13, $0, $1, $2, $3) },
    { dispatch(14, $0, $1, $2, $3) }, { dispatch(15, $0, $1, $2, $3) },
    { dispatch(16, $0, $1, $2, $3) }, { dispatch(17, $0, $1, $2, $3) },
    { dispatch(18, $0, $1, $2, $3) }, { dispatch(19, $0, $1, $2, $3) },
    { dispatch(20, $0, $1, $2, $3) }, { dispatch(21, $0, $1, $2, $3) },
    { dispatch(22, $0, $1, $2, $3) }, { dispatch(23, $0, $1, $2, $3) },
    { dispatch(24, $0, $1, $2, $3) }, { dispatch(25, $0, $1, $2, $3) },
    { dispatch(26, $0, $1, $2, $3) }, { dispatch(27, $0, $1, $2, $3) },
    { dispatch(28, $0, $1, $2, $3) }, { dispatch(29, $0, $1, $2, $3) },
    { dispatch(30, $0, $1, $2, $3) }, { dispatch(31, $0, $1, $2, $3) },
  ]

  package static var capacity: Int { trampolines.count }

  private let slotIndex: Int
  private let lock = NSLock()
  private var unregistered = false

  private init(slotIndex: Int) {
    self.slotIndex = slotIndex
  }

  deinit {
    unregister()
  }

  /// Releases this registration's C callback slot. Safe to call multiple times.
  package func unregister() {
    let shouldFree = lock.withLock {
      defer { unregistered = true }
      return !unregistered
    }
    if shouldFree {
      PoolStorage.shared.free(slotIndex)
    }
  }

  /// Acquires a callback slot that decodes `CResult` synchronously during the C callback and
  /// delivers the `Result` asynchronously on a serial `DispatchQueue`.
  package static func register<CResult, SwiftResult: Sendable, CCallback>(
    queueLabelSuffix: String,
    resultType: CResult.Type = CResult.self,
    as callbackType: CCallback.Type = CCallback.self,
    decode: @escaping @Sendable (UnsafePointer<CResult>, MpImagePtr?, Int) throws -> SwiftResult,
    handler: @escaping @Sendable (Result<SwiftResult, any Error>, Int) -> Void
  ) throws -> (registration: VisionLiveStreamRegistration, cCallback: CCallback) {
    precondition(MemoryLayout<CCallback>.size == MemoryLayout<RawCCallback>.size)
    let queue = DispatchQueue(
      label: "com.mediapipe.tasks.vision.\(queueLabelSuffix).\(UUID().uuidString)"
    )
    let slotIndex = try PoolStorage.shared.allocate { status, rawResult, cImage, timestampMs in
      let timestamp = Int(clamping: timestampMs)
      let outcome: Result<SwiftResult, any Error>
      if status != kMpOk {
        outcome = .failure(TasksError(code: TasksErrorCode(cStatus: status)))
      } else if let rawResult {
        outcome = Result {
          try decode(rawResult.assumingMemoryBound(to: CResult.self), cImage, timestamp)
        }
      } else {
        outcome = .failure(
          TasksError.internalError("Live-stream callback received kMpOk with a null result.")
        )
      }
      queue.async { handler(outcome, timestamp) }
    }
    return (
      registration: VisionLiveStreamRegistration(slotIndex: slotIndex),
      cCallback: unsafeBitCast(trampolines[slotIndex], to: callbackType)
    )
  }
}
