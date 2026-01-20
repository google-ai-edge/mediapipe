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
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult

/**
 * Internal settings for MediaPipe ObjectDetector task.
 *
 * @property displayNamesLocale The locale to use for display names.
 * @property maxResults The maximum number of top-scored detection results to return.
 * @property scoreThreshold The score threshold to filter out low-scoring results.
 * @property categoryAllowlist A list of allowed category names.
 * @property categoryDenylist A list of denied category names.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class ObjectDetectorSettingsInternal
constructor(
  val displayNamesLocale: String,
  val maxResults: Int,
  val scoreThreshold: Float,
  val categoryAllowlist: List<String>,
  val categoryDenylist: List<String>,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<ObjectDetectorResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to an [ObjectDetector.ObjectDetectorOptions].
   *
   * @param baseOptions The base options to use.
   * @return An [ObjectDetector.ObjectDetectorOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): ObjectDetector.ObjectDetectorOptions {
    val optionsBuilder =
      ObjectDetector.ObjectDetectorOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setDisplayNamesLocale(displayNamesLocale)
        .setScoreThreshold(scoreThreshold)
        .setCategoryAllowlist(categoryAllowlist)
        .setCategoryDenylist(categoryDenylist)
    if (maxResults > 0) {
      optionsBuilder.setMaxResults(maxResults)
    }
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
