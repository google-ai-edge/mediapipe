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
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizerResult

/**
 * Internal settings for MediaPipe GestureRecognizer task.
 *
 * @property minHandDetectionConfidence The minimum confidence score for hand detection to be
 *   considered successful.
 * @property minHandPresenceConfidence The minimum confidence score for hand presence to be
 *   considered successful.
 * @property minTrackingConfidence The minimum confidence score for hand tracking to be considered
 *   successful.
 * @property numHands The maximum number of hands to detect.
 * @property cannedGesturesClassifierSettings Optional options for the canned gestures classifier.
 * @property customGesturesClassifierSettings Optional options for the custom gestures classifier.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class GestureRecognizerSettingsInternal
constructor(
  val minHandDetectionConfidence: Float,
  val minHandPresenceConfidence: Float,
  val minTrackingConfidence: Float,
  val numHands: Int,
  val cannedGesturesClassifierSettings: ClassifierSettingsInternal?,
  val customGesturesClassifierSettings: ClassifierSettingsInternal?,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<GestureRecognizerResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to a [GestureRecognizer.GestureRecognizerOptions].
   *
   * @param baseOptions The base options to use.
   * @return A [GestureRecognizer.GestureRecognizerOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): GestureRecognizer.GestureRecognizerOptions {
    val optionsBuilder =
      GestureRecognizer.GestureRecognizerOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setNumHands(numHands)
        .setMinHandDetectionConfidence(minHandDetectionConfidence)
        .setMinHandPresenceConfidence(minHandPresenceConfidence)
        .setMinTrackingConfidence(minTrackingConfidence)
    cannedGesturesClassifierSettings?.let {
      optionsBuilder.setCannedGesturesClassifierOptions(it.toClassifierOptions())
    }
    customGesturesClassifierSettings?.let {
      optionsBuilder.setCustomGesturesClassifierOptions(it.toClassifierOptions())
    }
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
