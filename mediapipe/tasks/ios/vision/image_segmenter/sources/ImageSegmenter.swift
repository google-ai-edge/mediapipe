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
package import third_party_mediapipe_tasks_c_core_common
package import third_party_mediapipe_tasks_c_core_mp_status
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_core_image_processing_options
package import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_lib
package import third_party_mediapipe_tasks_c_vision_image_segmenter_image_segmenter_result

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Segments images, video frames, or live camera streams with a TFLite model that has metadata
/// (region-of-interest is unsupported).
///
/// - Note: This class is not thread-safe. Concurrent callers should serialize access (for example,
///   by isolating an instance inside a Swift `actor`).
public final class ImageSegmenter {
  /// Category labels from the model metadata, indexed by category mask value and confidence mask
  /// position, or empty if the model has no label map.
  public let labels: [String]

  private let cSegmenter: MpImageSegmenterPtr
  private let runner: VisionTaskRunner
  private let liveStreamRegistration: VisionLiveStreamRegistration?

  /// Creates an image segmenter and initializes the underlying MediaPipe task graph.
  ///
  /// - Throws: `TasksError` if `options` are inconsistent or graph initialization fails.
  public init(options: ImageSegmenterOptions) throws {
    let runner = try VisionTaskRunner(
      baseOptions: options.baseOptions, runningMode: options.runningMode, roiAllowed: false,
      hasLiveStreamHandler: options.liveStreamResultHandler != nil)
    let registered = try options.liveStreamResultHandler.map { handler in
      try VisionLiveStreamRegistration.register(
        queueLabelSuffix: "imageSegmenter", resultType: MpImageSegmenterResult.self,
        as: MpImageSegmenterResultCallback.self,
        decode: { cResult, _, timestamp in
          try ImageSegmenterResult(cResult: cResult.pointee, timestampInMilliseconds: timestamp)
        },
        handler: handler)
    }
    let cSegmenter: MpImageSegmenterPtr
    let labels: [String]
    do {
      cSegmenter = try options.withCOptions(resultCallback: registered?.cCallback) { cOptions in
        var segmenterPtr: MpImageSegmenterPtr?
        try MpStatus.check(defaultMessage: "Failed to create ImageSegmenter.") { errorMessage in
          MpImageSegmenterCreate(&cOptions, &segmenterPtr, &errorMessage)
        }
        guard let segmenterPtr else {
          throw TasksError.internalError("MpImageSegmenterCreate returned a null pointer.")
        }
        return segmenterPtr
      }
      do {
        labels = try Self.labels(of: cSegmenter)
      } catch {
        _ = MpImageSegmenterClose(cSegmenter, nil)
        throw error
      }
    } catch {
      registered?.registration.unregister()
      throw error
    }
    self.cSegmenter = cSegmenter
    self.labels = labels
    self.runner = runner
    self.liveStreamRegistration = registered?.registration
  }

  /// Creates an image segmenter in `.image` mode from a local model file path.
  ///
  /// - Throws: `TasksError` if the model cannot be loaded or initialized.
  public convenience init(modelPath: String) throws {
    try self.init(
      options: ImageSegmenterOptions(baseOptions: BaseOptions(modelAssetPath: modelPath)))
  }

  deinit {
    _ = MpImageSegmenterClose(cSegmenter, nil)
    liveStreamRegistration?.unregister()
  }

  /// Segments a single image in `.image` running mode.
  public func segment(
    image: MPImage,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageSegmenterResult {
    try runSegmentation(image, imageProcessingOptions, mode: .image, timestamp: 0) {
      MpImageSegmenterSegmentImage(cSegmenter, $0, $1, $2, &$3)
    }
  }

  /// Segments a decoded video frame at monotonically increasing `timestampInMilliseconds` in
  /// `.video` running mode.
  public func segment(
    videoFrame: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws -> ImageSegmenterResult {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    return try runSegmentation(
      videoFrame, imageProcessingOptions, mode: .video, timestamp: timestampInMilliseconds
    ) {
      MpImageSegmenterSegmentForVideo(cSegmenter, $0, $1, cTimestamp, $2, &$3)
    }
  }

  /// Sends a live-stream frame at monotonically increasing `timestampInMilliseconds` for
  /// asynchronous segmentation in `.liveStream` running mode.
  ///
  /// Results are delivered to `ImageSegmenterOptions.liveStreamResultHandler`.
  public func segmentAsync(
    image: MPImage,
    timestampInMilliseconds: Int,
    imageProcessingOptions: ImageProcessingOptions? = nil
  ) throws {
    let cTimestamp = try runner.cTimestamp(timestampInMilliseconds)
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: .liveStream
    ) { cImage, cOptions in
      try MpStatus.check(defaultMessage: "Failed to send frame to ImageSegmenter.") {
        errorMessage in
        MpImageSegmenterSegmentAsync(cSegmenter, cImage, cOptions, cTimestamp, &errorMessage)
      }
    }
  }

  private func runSegmentation(
    _ image: MPImage,
    _ imageProcessingOptions: ImageProcessingOptions?,
    mode: VisionRunningMode,
    timestamp: Int,
    _ call: (
      MpImagePtr, UnsafePointer<MpImageProcessingOptions>,
      UnsafeMutablePointer<MpImageSegmenterResult>, inout UnsafeMutablePointer<CChar>?
    ) -> MpStatus
  ) throws -> ImageSegmenterResult {
    var cResult = MpImageSegmenterResult()
    defer { MpImageSegmenterCloseResult(&cResult) }
    try runner.withPreparedInput(
      image: image, imageProcessingOptions: imageProcessingOptions, expectedMode: mode
    ) { cImage, cOptions in
      try MpStatus.check(defaultMessage: "Failed to segment image.") { errorMessage in
        call(cImage, cOptions, &cResult, &errorMessage)
      }
    }
    return try ImageSegmenterResult(cResult: cResult, timestampInMilliseconds: timestamp)
  }

  private static func labels(of cSegmenter: MpImageSegmenterPtr) throws -> [String] {
    var cLabels = MpStringList()
    defer { MpStringListFree(&cLabels) }
    try MpStatus.check(defaultMessage: "Failed to get ImageSegmenter labels.") { errorMessage in
      MpImageSegmenterGetLabels(cSegmenter, &cLabels, &errorMessage)
    }
    guard let cStrings = cLabels.strings else { return [] }
    return (0..<Int(cLabels.num_strings)).map { cStrings[$0].map { String(cString: $0) } ?? "" }
  }
}
