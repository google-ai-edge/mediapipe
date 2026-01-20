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
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

/**
 * Internal settings for MediaPipe FaceLandmarker task.
 *
 * @property minFaceDetectionConfidence The minimum confidence score for face detection to be
 *   considered successful.
 * @property minFacePresenceConfidence The minimum confidence score for face presence to be
 *   considered successful.
 * @property minTrackingConfidence The minimum confidence score for face tracking to be considered
 *   successful.
 * @property numFaces The maximum number of faces to detect.
 * @property outputFaceBlendshapes Whether to output face blendshapes.
 * @property outputFacialTransformationMatrixes Whether to output facial transformation matrixes.
 * @property runningMode The running mode of the task.
 * @property resultListener The listener to receive results.
 * @property errorListener The listener to receive errors.
 */
data class FaceLandmarkerSettingsInternal
constructor(
  val minFaceDetectionConfidence: Float,
  val minFacePresenceConfidence: Float,
  val minTrackingConfidence: Float,
  val numFaces: Int,
  val outputFaceBlendshapes: Boolean,
  val outputFacialTransformationMatrixes: Boolean,
  val runningMode: RunningMode,
  val resultListener: OutputHandler.ResultListener<FaceLandmarkerResult, MPImage>?,
  val errorListener: ErrorListener?,
) {
  /**
   * Converts the internal settings to a [FaceLandmarker.FaceLandmarkerOptions].
   *
   * @param baseOptions The base options to use.
   * @return A [FaceLandmarker.FaceLandmarkerOptions] instance.
   */
  fun toOptions(baseOptions: BaseOptions): FaceLandmarker.FaceLandmarkerOptions {
    val optionsBuilder =
      FaceLandmarker.FaceLandmarkerOptions.builder()
        .setBaseOptions(baseOptions)
        .setRunningMode(runningMode)
        .setNumFaces(numFaces)
        .setMinFaceDetectionConfidence(minFaceDetectionConfidence)
        .setMinFacePresenceConfidence(minFacePresenceConfidence)
        .setMinTrackingConfidence(minTrackingConfidence)
        .setOutputFaceBlendshapes(outputFaceBlendshapes)
        .setOutputFacialTransformationMatrixes(outputFacialTransformationMatrixes)
    resultListener?.let { optionsBuilder.setResultListener(it) }
    errorListener?.let { optionsBuilder.setErrorListener(it) }
    return optionsBuilder.build()
  }
}
