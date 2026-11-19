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

/// Receives the live-stream results of one Vision task from the task's C result callback.
///
/// The task sets `userData` as `result_callback_user_data` in its C options, and sets
/// `result_callback` to a `@convention(c)` closure that forwards its arguments to
/// `receive(userData:status:errorMessage:result:image:timestampMs:)`:
///
/// ```swift
/// cOptions.result_callback = {
///   VisionLiveStreamContext.receive(
///     userData: $0, status: $1, errorMessage: $2, result: $3, image: $4, timestampMs: $5)
/// }
/// cOptions.result_callback_user_data = context.userData
/// ```
///
/// The C task does not retain `userData`, so the Swift task must keep this object alive until its
/// `Mp*Close` call has returned. `Mp*Close` waits for pending results and invokes no callback after
/// it returns. Code that does not hold the context in a stored property during such a C call must
/// wrap the call in `withExtendedLifetime`.
///
/// Decodes `CResult` synchronously before the C pointers expire and delivers the outcome on a
/// private serial queue.
package final class VisionLiveStreamContext<CResult>: Sendable {
  private let deliver:
    @Sendable (MpStatus, UnsafePointer<CChar>?, UnsafePointer<CResult>?, MpImagePtr?, Int64) -> Void

  /// Creates a context that decodes each `CResult` with `decode` and passes the outcome and the
  /// timestamp in milliseconds to `handler` on a serial `DispatchQueue`.
  package init<SwiftResult: Sendable>(
    queueLabelSuffix: String,
    decode: @escaping @Sendable (UnsafePointer<CResult>, MpImagePtr?, Int) throws -> SwiftResult,
    handler: @escaping @Sendable (Result<SwiftResult, any Error>, Int) -> Void
  ) {
    let queue = DispatchQueue(
      label: "com.mediapipe.tasks.vision.\(queueLabelSuffix).\(UUID().uuidString)"
    )
    deliver = { status, errorMessage, cResult, cImage, timestampMs in
      let timestamp = Int(clamping: timestampMs)
      let outcome: Result<SwiftResult, any Error>
      if status != kMpOk {
        outcome = .failure(
          TasksError(
            code: TasksErrorCode(cStatus: status),
            message: errorMessage.map { String(cString: $0) }
          )
        )
      } else if let cResult {
        outcome = Result {
          try decode(cResult, cImage, timestamp)
        }
      } else {
        outcome = .failure(
          TasksError.internalError("Live-stream callback received kMpOk with a null result.")
        )
      }
      queue.async { handler(outcome, timestamp) }
    }
  }

  /// The value for `result_callback_user_data` in the C options. Does not retain `self`.
  package var userData: UnsafeMutableRawPointer {
    Unmanaged.passUnretained(self).toOpaque()
  }

  /// Forwards the arguments of a C result callback to the context that `userData` points to.
  package static func receive(
    userData: UnsafeMutableRawPointer?,
    status: MpStatus,
    errorMessage: UnsafePointer<CChar>?,
    result: UnsafePointer<CResult>?,
    image: MpImagePtr?,
    timestampMs: Int64
  ) {
    guard let userData else { return }
    Unmanaged<VisionLiveStreamContext<CResult>>.fromOpaque(userData).takeUnretainedValue()
      .deliver(status, errorMessage, result, image, timestampMs)
  }
}
