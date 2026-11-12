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

/** Scoring mode for multi-token choice and rubric evaluation. */
enum class ScoringMode(val value: Int) {
  /** Length-normalized log-likelihood across option tokens (default). */
  LENGTH_NORMALIZED_LOG_PROB(0),
  /** Unnormalized sum of token log-likelihoods. */
  SUM_LOG_PROB(1),
  /** First-token logit comparison when all choice keys have distinct leading tokens. */
  FIRST_TOKEN_ONLY(2),
}

/** How examples and inputs are embedded by embedding models, for questions with examples. */
enum class InputPrompt(val value: Int) {
  /** `"task: classification | query: <text>"`. Best for sentences (default). */
  CLASSIFICATION(0),
  /** The raw text. Best for single words and short phrases. */
  NONE(1),
}

/** Sealed hierarchy of structured questions evaluated by [Decision]. */
sealed interface DecisionQuestion

/**
 * Evaluates a natural-language boolean predicate as `true` or `false` with a calibrated probability
 * and confidence score.
 *
 * @property condition Natural-language statement to evaluate (e.g. `"The user wants to cancel"`).
 * @property threshold Probability threshold in `[0, 1]` above which [BooleanResult.value] is true.
 * @property temperature Softmax temperature override (`0f` uses the model's calibrated default).
 * @property normalizePrior Whether to subtract the empty-context prior log-odds before
 *   thresholding.
 * @property positiveExamples Example inputs for which the answer is true. With [negativeExamples],
 *   embedding models fit a classifier on them and [condition] may be blank.
 * @property negativeExamples Example inputs for which the answer is false.
 * @property inputPrompt How examples and inputs are embedded. Only used with examples.
 */
data class BooleanQuestion
@JvmOverloads
constructor(
  val condition: String = "",
  val threshold: Float = 0.5f,
  val temperature: Float = 0.0f,
  val normalizePrior: Boolean = false,
  val positiveExamples: List<String> = emptyList(),
  val negativeExamples: List<String> = emptyList(),
  val inputPrompt: InputPrompt = InputPrompt.CLASSIFICATION,
) : DecisionQuestion {
  init {
    require(positiveExamples.isEmpty() == negativeExamples.isEmpty()) {
      "A BooleanQuestion with examples needs both positiveExamples and negativeExamples."
    }
    require(condition.isNotBlank() || positiveExamples.isNotEmpty()) {
      "BooleanQuestion.condition must not be blank."
    }
    require(threshold in 0.0f..1.0f) {
      "BooleanQuestion.threshold must be in [0.0, 1.0], got $threshold."
    }
  }
}

/**
 * Evaluates a mutually exclusive categorical choice menu and returns a calibrated probability
 * distribution over [choices].
 *
 * @property choices Ordered list of option keys.
 * @property descriptions Optional descriptions corresponding 1:1 with [choices].
 * @property temperature Softmax temperature override (`0f` uses the model's calibrated default).
 * @property instructions Optional task instructions prepended to the choice prompt.
 * @property scoringMode Multi-token log-probability aggregation strategy.
 * @property normalizePrior Whether to normalize option log-probabilities against the empty-context
 *   prior.
 * @property examples Optional example inputs, one list per choice in the order of [choices]. If any
 *   choice has examples, every choice needs at least one, and embedding models fit a classifier on
 *   them; all-empty lists mean no examples.
 * @property inputPrompt How examples and inputs are embedded. Only used with examples.
 * @property balanceClassWeights With [examples], weights each choice by `N / (K * N_c)`, so that
 *   choices with fewer examples aren't penalized.
 */
data class ChoiceQuestion
@JvmOverloads
constructor(
  val choices: List<String>,
  val descriptions: List<String> = emptyList(),
  val temperature: Float = 0.0f,
  val instructions: String = "",
  val scoringMode: ScoringMode = ScoringMode.LENGTH_NORMALIZED_LOG_PROB,
  val normalizePrior: Boolean = false,
  val examples: List<List<String>> = emptyList(),
  val inputPrompt: InputPrompt = InputPrompt.CLASSIFICATION,
  val balanceClassWeights: Boolean = true,
) : DecisionQuestion {
  /**
   * Ordered map of choice keys to semantic descriptions (empty strings when constructed from
   * [choices] without descriptions).
   */
  val criteria: Map<String, String> by lazy {
    buildMap(choices.size) {
      for (i in choices.indices) {
        put(choices[i], descriptions.getOrElse(i) { "" })
      }
    }
  }

  /**
   * Creates a [ChoiceQuestion] from ordered key-description [criteria] pairs (iteration order of
   * [criteria] is preserved, e.g. from `mapOf` or `LinkedHashMap`).
   */
  @JvmOverloads
  constructor(
    criteria: Map<String, String>,
    temperature: Float = 0.0f,
    instructions: String = "",
    scoringMode: ScoringMode = ScoringMode.LENGTH_NORMALIZED_LOG_PROB,
    normalizePrior: Boolean = false,
  ) : this(
    choices = criteria.keys.toList(),
    descriptions =
      if (criteria.values.all { it.isEmpty() }) {
        emptyList()
      } else {
        criteria.values.toList()
      },
    temperature = temperature,
    instructions = instructions,
    scoringMode = scoringMode,
    normalizePrior = normalizePrior,
  )

  init {
    require(choices.isNotEmpty()) { "ChoiceQuestion.choices must not be empty." }
    if (descriptions.isNotEmpty()) {
      require(descriptions.size == choices.size) {
        "ChoiceQuestion.descriptions size (${descriptions.size}) must match choices size (${choices.size})."
      }
    }
    if (hasExamples) {
      require(choices.size >= 2) { "A ChoiceQuestion with examples needs at least 2 choices." }
      require(examples.size == choices.size && examples.all { it.isNotEmpty() }) {
        "Every choice needs at least one example."
      }
    }
  }

  /** Whether any choice has an example. As in the C API, all-empty lists mean no examples. */
  internal val hasExamples: Boolean
    get() = examples.any { it.isNotEmpty() }

  companion object {
    @Deprecated(
      "Use ChoiceQuestion(criteria = ...) constructor instead.",
      ReplaceWith("ChoiceQuestion(options, temperature, instructions, scoringMode, normalizePrior)"),
    )
    @JvmStatic
    @JvmOverloads
    fun withDescriptions(
      options: Map<String, String>,
      temperature: Float = 0.0f,
      instructions: String = "",
      scoringMode: ScoringMode = ScoringMode.LENGTH_NORMALIZED_LOG_PROB,
      normalizePrior: Boolean = false,
    ): ChoiceQuestion =
      ChoiceQuestion(
        criteria = options,
        temperature = temperature,
        instructions = instructions,
        scoringMode = scoringMode,
        normalizePrior = normalizePrior,
      )

    /**
     * Creates a [ChoiceQuestion] from choice keys mapped to example inputs, with optional
     * [descriptions] (choice key to description) for some or all keys. Requires at least 2 keys,
     * each with at least one example.
     */
    @JvmStatic
    fun withExamples(
      examples: Map<String, List<String>>,
      descriptions: Map<String, String> = emptyMap(),
      inputPrompt: InputPrompt = InputPrompt.CLASSIFICATION,
      balanceClassWeights: Boolean = true,
      temperature: Float = 0.0f,
      instructions: String = "",
    ): ChoiceQuestion {
      require(examples.size >= 2 && examples.values.all { it.isNotEmpty() }) {
        "withExamples needs at least 2 choices, each with at least one example."
      }
      val keys = examples.keys.toList()
      return ChoiceQuestion(
        choices = keys,
        descriptions =
          if (descriptions.isEmpty()) emptyList() else keys.map { descriptions[it] ?: "" },
        temperature = temperature,
        instructions = instructions,
        examples = keys.map { examples.getValue(it) },
        inputPrompt = inputPrompt,
        balanceClassWeights = balanceClassWeights,
      )
    }
  }
}

/**
 * Evaluates an ordered rubric of levels and computes both the most likely rubric level and a
 * continuous expected score in `[0.0, K - 1]` across the `K` rubric levels.
 *
 * @property rubric Ordered list of `K` rubric levels from lowest (`0`) to highest (`K - 1`).
 * @property temperature Softmax temperature override (`0f` uses the model's calibrated default).
 * @property instructions Optional task instructions prepended to the rubric prompt.
 */
data class ScoreQuestion(
  val rubric: List<String>,
  val temperature: Float = 0.0f,
  val instructions: String = "",
) : DecisionQuestion {
  init {
    require(rubric.size >= 2) {
      "ScoreQuestion.rubric must contain at least 2 ordered levels, got ${rubric.size}."
    }
  }
}

/**
 * Perceptual input context for multimodal or text-only decision evaluation.
 *
 * @property text Primary text context or transcript.
 * @property imageRgb Raw interleaved RGB/RGBA pixel bytes (`width * height * channels`).
 * @property imageWidth Width of [imageRgb] in pixels.
 * @property imageHeight Height of [imageRgb] in pixels.
 * @property imageChannels Channel count of [imageRgb] (3 for RGB, 4 for RGBA).
 * @property imageBytes Encoded PNG/JPEG image bytes.
 * @property audioSamples Mono float32 PCM audio samples in `[-1.0, 1.0]`.
 * @property audioSampleRate Sample rate in Hz for [audioSamples] (defaults to 16000).
 * @property embedding Optional precomputed context embedding vector (supported only by
 *   embedding-based models such as EmbeddingGemma).
 */
data class DecisionContext(
  val text: String = "",
  val imageRgb: ByteArray? = null,
  val imageWidth: Int = 0,
  val imageHeight: Int = 0,
  val imageChannels: Int = 3,
  val imageBytes: ByteArray? = null,
  val audioSamples: FloatArray? = null,
  val audioSampleRate: Int = 16000,
  val embedding: FloatArray? = null,
)
