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
package import third_party_mediapipe_tasks_c_vision_image_classifier_image_classifier_lib

#if !MEDIAPIPE_SINGLE_MODULE
  @_exported import MediaPipeTasksComponents
  @_exported import MediaPipeTasksCore
  @_exported import MediaPipeTasksVisionCore
#endif

/// Output of an `ImageClassifier` inference pass on a single image or video frame.
public struct ImageClassifierResult: Sendable, Equatable, Hashable {
  /// Predictions produced by each classification head of the model.
  public let classificationResult: ClassificationResult

  /// Frame timestamp in milliseconds, or `0` in `.image` mode.
  public let timestampInMilliseconds: Int

  /// Creates a result from head predictions and the associated frame timestamp.
  public init(classificationResult: ClassificationResult, timestampInMilliseconds: Int) {
    self.classificationResult = classificationResult
    self.timestampInMilliseconds = timestampInMilliseconds
  }

  package init(cResult: MpImageClassifierResult, timestampInMilliseconds: Int) {
    self.classificationResult = ClassificationResult(cResult: cResult)
    self.timestampInMilliseconds = timestampInMilliseconds
  }
}
