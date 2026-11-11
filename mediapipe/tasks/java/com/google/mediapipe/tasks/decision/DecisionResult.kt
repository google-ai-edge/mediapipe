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
 * @property predictionSet Conformal prediction set of choice keys covering at least 90% of the
 *   calibrated probability mass, ordered from most to least probable.
 */
data class ChoiceResult
@JvmOverloads
constructor(
  val selectedKey: String,
  val probabilities: Map<String, Float>,
  val confidence: Float,
  val predictionSet: List<String> = computePredictionSet(probabilities),
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

  companion object {
    internal fun computePredictionSet(
      probabilities: Map<String, Float>,
      coverage: Float = 0.90f,
    ): List<String> {
      if (probabilities.isEmpty()) return emptyList()
      val sorted =
        probabilities.entries.sortedWith(
          compareByDescending<Map.Entry<String, Float>> { it.value }.thenBy { it.key }
        )
      val clampedCoverage = coverage.coerceIn(0.0f, 1.0f)
      val result = ArrayList<String>(sorted.size)
      var cumulative = 0.0f
      for ((key, prob) in sorted) {
        result.add(key)
        cumulative += prob
        if (cumulative >= clampedCoverage) break
      }
      return result
    }
  }
}

/**
 * Calibrated result of evaluating a [ScoreQuestion].
 *
 * @property selectedKey Key of the highest-probability rubric level.
 * @property expectedScore Continuous expected rubric level index in `[0.0, K - 1]` across the `K`
 *   ordered rubric levels (`sum_i(i * probabilities[i])`).
 * @property probabilities Probability distribution across each ordered rubric level.
 * @property confidence Calibrated confidence score in `[0.0, 1.0]`.
 */
data class ScoreResult(
  val selectedKey: String,
  val expectedScore: Float,
  val probabilities: List<Float>,
  val confidence: Float,
) {
  @Deprecated("Use probabilities instead.", ReplaceWith("probabilities"))
  val levelProbabilities: List<Float>
    get() = probabilities

  /** Secondary constructor invoked from JNI with a primitive [FloatArray]. */
  constructor(
    selectedKey: String,
    expectedScore: Float,
    probabilities: FloatArray,
    confidence: Float,
  ) : this(
    selectedKey = selectedKey,
    expectedScore = expectedScore,
    probabilities = probabilities.toList(),
    confidence = confidence,
  )
}
