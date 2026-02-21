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

package com.google.mediapipe.tasks.vision.provider

import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.ErrorListener
import com.google.mediapipe.tasks.core.OutputHandler
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult

/**
 * Internal settings for MediaPipe FaceDetector task.
 *
 * @property minDetectionConfidence The minimum confidence score for face detection.
 * @property minSuppressionThreshold The minimum suppression threshold.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class FaceDetectorSettingsInternal
constructor(
  val minDetectionConfidence: Float,
  val minSuppressionThreshold: Float,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<FaceDetectorResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to a [FaceDetector.FaceDetectorOptions].
   *
   * @param baseOptions The base options to use.
   * @return A [FaceDetector.FaceDetectorOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): FaceDetector.FaceDetectorOptions {
    val optionsBuilder =
      FaceDetector.FaceDetectorOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setMinDetectionConfidence(minDetectionConfidence)
        .setMinSuppressionThreshold(minSuppressionThreshold)
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
