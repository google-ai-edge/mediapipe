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
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Internal settings for MediaPipe PoseLandmarker task.
 *
 * @property minPoseDetectionConfidence The minimum confidence score for pose detection to be
 *   considered successful.
 * @property minPosePresenceConfidence The minimum confidence score for pose presence to be
 *   considered successful.
 * @property minTrackingConfidence The minimum confidence score for pose tracking to be considered
 *   successful.
 * @property numPoses The maximum number of poses to detect.
 * @property outputSegmentationMasks Whether to output segmentation masks.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class PoseLandmarkerSettingsInternal
constructor(
  val minPoseDetectionConfidence: Float,
  val minPosePresenceConfidence: Float,
  val minTrackingConfidence: Float,
  val numPoses: Int,
  val outputSegmentationMasks: Boolean,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<PoseLandmarkerResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to a [PoseLandmarker.PoseLandmarkerOptions].
   *
   * @param baseOptions The base options to use.
   * @return A [PoseLandmarker.PoseLandmarkerOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): PoseLandmarker.PoseLandmarkerOptions {
    val optionsBuilder =
      PoseLandmarker.PoseLandmarkerOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setNumPoses(numPoses)
        .setMinPoseDetectionConfidence(minPoseDetectionConfidence)
        .setMinPosePresenceConfidence(minPosePresenceConfidence)
        .setMinTrackingConfidence(minTrackingConfidence)
        .setOutputSegmentationMasks(outputSegmentationMasks)
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
