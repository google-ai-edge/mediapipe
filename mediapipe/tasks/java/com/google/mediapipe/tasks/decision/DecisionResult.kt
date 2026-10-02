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

package com.google.mediapipe.tasks.decision

/**
 * Calibrated result of evaluating a [BooleanQuestion].
 *
 * @property value Whether [probabilityTrue] meets or exceeds the question's threshold.
 * @property probabilityTrue Calibrated probability in `[0.0, 1.0]` that the condition holds.
 * @property confidence Calibrated confidence score in `[0.0, 1.0]`.
 */
data class BooleanResult(val value: Boolean, val probabilityTrue: Float, val confidence: Float)

/**
 * Calibrated result of evaluating a [ChoiceQuestion].
 *
 * @property selectedKey Key of the highest-probability choice.
 * @property probabilities Map from each choice key to its calibrated probability in `[0.0, 1.0]`.
 * @property confidence Calibrated confidence score in `[0.0, 1.0]` for the selected choice.
 */
data class ChoiceResult(
  val selectedKey: String,
  val probabilities: Map<String, Float>,
  val confidence: Float,
) {
  /** Secondary constructor invoked from JNI with parallel key and probability arrays. */
  constructor(
    selectedKey: String,
    keys: Array<String>,
    probabilityValues: FloatArray,
    confidence: Float,
  ) : this(
    selectedKey = selectedKey,
    probabilities =
      buildMap(keys.size) {
        for (i in keys.indices) {
          put(keys[i], if (i < probabilityValues.size) probabilityValues[i] else 0.0f)
        }
      },
    confidence = confidence,
  )
}

/**
 * Calibrated result of evaluating a [ScoreQuestion].
 *
 * @property selectedKey Key of the highest-probability rubric level.
 * @property expectedScore Continuous expected score in `[0.0, 1.0]` across the ordered rubric
 *   levels.
 * @property levelProbabilities Probability distribution across each ordered rubric level.
 * @property confidence Calibrated confidence score in `[0.0, 1.0]`.
 */
data class ScoreResult(
  val selectedKey: String,
  val expectedScore: Float,
  val levelProbabilities: List<Float>,
  val confidence: Float,
) {
  /** Secondary constructor invoked from JNI with a primitive [FloatArray]. */
  constructor(
    selectedKey: String,
    expectedScore: Float,
    probabilities: FloatArray,
    confidence: Float,
  ) : this(
    selectedKey = selectedKey,
    expectedScore = expectedScore,
    levelProbabilities = probabilities.toList(),
    confidence = confidence,
  )
}
