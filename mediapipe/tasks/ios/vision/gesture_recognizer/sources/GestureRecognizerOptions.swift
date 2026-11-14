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
package import third_party_mediapipe_tasks_c_components_processors_classifier_options
package import third_party_mediapipe_tasks_c_core_base_options
package import third_party_mediapipe_tasks_c_vision_core_image
package import third_party_mediapipe_tasks_c_vision_gesture_recognizer_gesture_recognizer_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Configuration for initializing a `GestureRecognizer`.
///
/// Not `Equatable` or `Hashable` because `liveStreamResultHandler` is a closure.
public struct GestureRecognizerOptions: Sendable {
  /// Asynchronous callback receiving live-stream output and the frame timestamp in milliseconds.
  public typealias LiveStreamResultHandler =
    @Sendable (Result<GestureRecognizerResult, any Error>, _ timestampInMilliseconds: Int) -> Void

  /// Model asset loading and hardware acceleration settings.
  public var baseOptions: BaseOptions
  /// Execution mode (`.image`, `.video`, or `.liveStream`).
  public var runningMode: VisionRunningMode
  /// Maximum number of hands detected per frame (defaults to `1`).
  public var numHands: Int
  /// Minimum confidence score (`[0.0, 1.0]`) for palm detection (defaults to `0.5`).
  public var minHandDetectionConfidence: Float
  /// Minimum hand-presence confidence score (`[0.0, 1.0]`) in landmark regression (`0.5`).
  public var minHandPresenceConfidence: Float
  /// Minimum landmark tracking confidence (`[0.0, 1.0]`) across video/stream frames (`0.5`).
  public var minTrackingConfidence: Float
  /// Threshold and allowlist/denylist settings for built-in canned gestures.
  public var cannedGesturesClassifierOptions: ClassifierOptions
  /// Threshold and allowlist/denylist settings for Model Maker custom gestures.
  public var customGesturesClassifierOptions: ClassifierOptions
  /// Callback required when `runningMode == .liveStream` and disallowed in other modes.
  public var liveStreamResultHandler: LiveStreamResultHandler?

  /// Creates configuration options defaulting to `.image` mode and single-hand detection.
  public init(
    baseOptions: BaseOptions = BaseOptions(), runningMode: VisionRunningMode = .image,
    numHands: Int = 1, minHandDetectionConfidence: Float = 0.5,
    minHandPresenceConfidence: Float = 0.5, minTrackingConfidence: Float = 0.5,
    cannedGesturesClassifierOptions: ClassifierOptions = ClassifierOptions(),
    customGesturesClassifierOptions: ClassifierOptions = ClassifierOptions(),
    liveStreamResultHandler: LiveStreamResultHandler? = nil
  ) {
    self.baseOptions = baseOptions
    self.runningMode = runningMode
    self.numHands = numHands
    self.minHandDetectionConfidence = minHandDetectionConfidence
    self.minHandPresenceConfidence = minHandPresenceConfidence
    self.minTrackingConfidence = minTrackingConfidence
    self.cannedGesturesClassifierOptions = cannedGesturesClassifierOptions
    self.customGesturesClassifierOptions = customGesturesClassifierOptions
    self.liveStreamResultHandler = liveStreamResultHandler
  }

  package func withCOptions<Result>(
    cCallback: MpGestureRecognizerResultCallback?,
    _ body: (inout MpGestureRecognizerOptions) throws -> Result
  ) throws -> Result {
    guard numHands > 0, let cNumHands = Int32(exactly: numHands) else {
      throw TasksError.invalidArgument("numHands (\(numHands)) must be a positive Int32.")
    }
    return try baseOptions.withCOptions { cBase in
      try cannedGesturesClassifierOptions.withCOptions { cCanned in
        try customGesturesClassifierOptions.withCOptions { cCustom in
          var cOptions = MpGestureRecognizerOptions(
            base_options: cBase, running_mode: runningMode.cRunningMode,
            num_hands: cNumHands,
            min_hand_detection_confidence: minHandDetectionConfidence,
            min_hand_presence_confidence: minHandPresenceConfidence,
            min_tracking_confidence: minTrackingConfidence,
            canned_gestures_classifier_options: cCanned,
            custom_gestures_classifier_options: cCustom, result_callback: cCallback)
          return try body(&cOptions)
        }
      }
    }
  }
}
