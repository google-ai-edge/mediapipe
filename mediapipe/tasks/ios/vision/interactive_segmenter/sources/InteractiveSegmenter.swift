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
import UIKit
package import third_party_mediapipe_tasks_c_core_base_options
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_interactive_segmenter_interactive_segmenter

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Performs interactive segmentation on images using a split encoder-decoder architecture.
///
/// `InteractiveSegmenter` splits inference into an image encoder (invoked once per image via
/// `set(image:)`) and a lightweight mask decoder (invoked per user stroke interaction via
/// `segment(strokes:)`).
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class InteractiveSegmenter {
  private let cSegmenter: MpInteractiveSegmenterPtr
  private var inputImage: MPImage?

  /// Creates an interactive segmenter and initializes the underlying MediaPipe task graph.
  ///
  /// - Throws: `TasksError` if the model asset is not specified or initialization fails.
  public init(options: InteractiveSegmenterOptions) throws {
    let hasModelPath = !options.baseOptions.modelAssetPath.isEmpty
    let hasModelBuffer = options.baseOptions.modelAssetBuffer.map { !$0.isEmpty } ?? false
    guard hasModelPath || hasModelBuffer else {
      throw TasksError.invalidArgument(
        "InteractiveSegmenterOptions.baseOptions must specify a non-empty modelAssetPath or "
          + "modelAssetBuffer."
      )
    }

    self.cSegmenter = try Self.createCSegmenter(baseOptions: options.baseOptions)
  }

  /// Creates an interactive segmenter from a `.task` model file path with default options.
  ///
  /// - Throws: `TasksError` if the model cannot be loaded.
  public convenience init(modelPath: String) throws {
    var options = InteractiveSegmenterOptions()
    options.baseOptions.modelAssetPath = modelPath
    try self.init(options: options)
  }

  deinit {
    _ = MpInteractiveSegmenterClose(cSegmenter, nil)
  }

  /// Encodes `image` for subsequent `segment(strokes:)` calls.
  public func set(image: MPImage) throws {
    try image.withCImage { cImage in
      try MpStatus.check(defaultMessage: "Failed to set image on InteractiveSegmenter.") {
        errorMessage in
        MpInteractiveSegmenterSetImage(cSegmenter, cImage, &errorMessage)
      }
    }
    self.inputImage = image
  }

  /// Segments the previously encoded image using `strokes`, returning a single-channel `VEC32F1`
  /// mask matching the dimensions and orientation of the input image.
  ///
  /// - Throws: `TasksError` if no image has been set, `strokes` is empty, or inference fails.
  public func segment(strokes: [Stroke]) throws -> MPImage {
    guard let inputImage else {
      throw TasksError.failedPrecondition(
        "Image not set - call set(image:) before segment(strokes:)."
      )
    }
    guard !strokes.isEmpty else {
      throw TasksError.invalidArgument("At least one stroke must be provided.")
    }
    guard let cStrokesCount = UInt32(exactly: strokes.count) else {
      throw TasksError.invalidArgument("Stroke count exceeds UInt32.max.")
    }

    var contiguousPoints: [MpStrokePoint] = []
    for stroke in strokes {
      let (newTotal, overflow) = contiguousPoints.count.addingReportingOverflow(stroke.points.count)
      guard !overflow, UInt32(exactly: stroke.points.count) != nil else {
        throw TasksError.invalidArgument("Stroke point count exceeds supported limits.")
      }
      contiguousPoints.reserveCapacity(newTotal)
      for keypoint in stroke.points {
        let mapped = Self.mapNormalizedPoint(keypoint.location, for: inputImage.orientation)
        contiguousPoints.append(MpStrokePoint(x: Float(mapped.x), y: Float(mapped.y)))
      }
    }

    var cMask: MpImagePtr?
    try contiguousPoints.withUnsafeMutableBufferPointer { pointsBuffer in
      var cStrokesArray: [MpStroke] = []
      cStrokesArray.reserveCapacity(strokes.count)
      var pointOffset = 0

      for stroke in strokes {
        let count = stroke.points.count
        let strokePointsPtr: UnsafeMutablePointer<MpStrokePoint>? =
          if count > 0, let baseAddress = pointsBuffer.baseAddress {
            baseAddress.advanced(by: pointOffset)
          } else {
            nil
          }
        pointOffset += count
        cStrokesArray.append(
          MpStroke(
            brush_mode: stroke.brushMode.cBrushMode,
            points: strokePointsPtr,
            points_count: UInt32(count),
            is_completed: stroke.isCompleted
          )
        )
      }

      try cStrokesArray.withUnsafeMutableBufferPointer { strokesBuffer in
        var cStrokes = MpStrokes(
          strokes: strokesBuffer.baseAddress,
          strokes_count: cStrokesCount
        )
        try MpStatus.check(defaultMessage: "Failed to run InteractiveSegmenter segmentation.") {
          errorMessage in
          MpInteractiveSegmenterSegment(
            cSegmenter,
            &cStrokes,
            &cMask,
            &errorMessage
          )
        }
      }
    }

    guard let validMask = cMask else {
      throw TasksError.internalError("MpInteractiveSegmenterSegment returned a null mask image.")
    }
    defer {
      MpImageFree(validMask)
    }

    return try MPImage(cMaskImage: validMask, orientation: inputImage.orientation)
  }

  /// Maps normalized `(x, y)` coordinates from display orientation into the underlying buffer's
  /// coordinate space.
  package static func mapNormalizedPoint(
    _ location: CGPoint,
    for orientation: UIImage.Orientation
  ) -> CGPoint {
    let x = location.x
    let y = location.y
    switch orientation {
    case .up:
      return CGPoint(x: x, y: y)
    case .right:
      return CGPoint(x: y, y: 1.0 - x)
    case .down:
      return CGPoint(x: 1.0 - x, y: 1.0 - y)
    case .left:
      return CGPoint(x: 1.0 - y, y: x)
    case .upMirrored:
      return CGPoint(x: 1.0 - x, y: y)
    case .downMirrored:
      return CGPoint(x: x, y: 1.0 - y)
    case .leftMirrored:
      return CGPoint(x: y, y: x)
    case .rightMirrored:
      return CGPoint(x: 1.0 - y, y: 1.0 - x)
    @unknown default:
      return CGPoint(x: x, y: y)
    }
  }

  // MARK: - Private C Bridging Helpers

  private static func createCSegmenter(
    baseOptions: BaseOptions
  ) throws -> MpInteractiveSegmenterPtr {
    try baseOptions.withCOptions { cBaseOptions in
      var cOptions = MpInteractiveSegmenterOptions(base_options: cBaseOptions)
      var segmenterPtr: MpInteractiveSegmenterPtr?
      try MpStatus.check(defaultMessage: "Failed to create InteractiveSegmenter.") { errorMessage in
        MpInteractiveSegmenterCreate(&cOptions, &segmenterPtr, &errorMessage)
      }
      guard let validSegmenter = segmenterPtr else {
        throw TasksError.internalError(
          "MpInteractiveSegmenterCreate returned a null pointer."
        )
      }
      return validSegmenter
    }
  }
}
