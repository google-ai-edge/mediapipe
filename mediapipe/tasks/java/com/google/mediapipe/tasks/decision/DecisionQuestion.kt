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
  /** Not set (default). Questions with examples use [CLASSIFICATION]. */
  UNSPECIFIED(0),
  /** `"task: classification | query: <text>"`. Best for sentences. */
  CLASSIFICATION(1),
  /** The raw text. Best for single words and short phrases. */
  NONE(2),
}

/** Sealed hierarchy of structured questions evaluated by [Decision]. */
sealed interface DecisionQuestion

/**
 * Evaluates a natural-language boolean predicate as `true` or `false` with a calibrated probability
 * and confidence score.
 *
 * @property condition Natural-language statement to evaluate (optional when examples or
 *   [fittedHead] are provided).
 * @property threshold Probability threshold in `[0, 1]` above which [BooleanResult.value] is true.
 * @property temperature Softmax temperature override (`0f` uses the model's calibrated default).
 * @property normalizePrior Whether to subtract the empty-context prior log-odds before
 *   thresholding.
 * @property positiveExamples Example inputs for which the answer is true. With [negativeExamples],
 *   embedding models fit a classifier on them and [condition] may be blank.
 * @property negativeExamples Example inputs for which the answer is false.
 * @property inputPrompt How examples and inputs are embedded. Only used with examples.
 * @param fittedHead Optional pre-fitted `FittedHead` binary payload for embedding models; copied on
 *   construction. Cannot be combined with examples.
 */
// Not a data class so that the fittedHead payload can be defensively copied in and out.
// fittedHead is an opaque binary payload: ByteArray avoids boxing every byte in a List<Byte>.
@Suppress("KotlinArray")
class BooleanQuestion
@JvmOverloads
constructor(
  val condition: String = "",
  val threshold: Float = 0.5f,
  val temperature: Float = 0.0f,
  val normalizePrior: Boolean = false,
  val positiveExamples: List<String> = emptyList(),
  val negativeExamples: List<String> = emptyList(),
  val inputPrompt: InputPrompt = InputPrompt.UNSPECIFIED,
  fittedHead: ByteArray? = null,
) : DecisionQuestion {
  private val fittedHeadBytes: ByteArray? = fittedHead?.copyOf()

  /** A copy of the optional pre-fitted `FittedHead` binary payload for embedding models. */
  val fittedHead: ByteArray?
    get() = fittedHeadBytes?.copyOf()

  init {
    require(positiveExamples.isEmpty() == negativeExamples.isEmpty()) {
      "A BooleanQuestion with examples needs both positiveExamples and negativeExamples."
    }
    if (fittedHeadBytes != null) {
      require(fittedHeadBytes.isNotEmpty()) { "BooleanQuestion.fittedHead must not be empty." }
      require(positiveExamples.isEmpty() && negativeExamples.isEmpty()) {
        "A BooleanQuestion cannot specify both fittedHead and examples."
      }
    }
    require(condition.isNotBlank() || positiveExamples.isNotEmpty() || fittedHeadBytes != null) {
      "BooleanQuestion.condition must not be blank unless examples or fittedHead are provided."
    }
    require(threshold in 0.0f..1.0f) {
      "BooleanQuestion.threshold must be in [0.0, 1.0], got $threshold."
    }
  }

  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is BooleanQuestion) return false
    return condition == other.condition &&
      threshold == other.threshold &&
      temperature == other.temperature &&
      normalizePrior == other.normalizePrior &&
      positiveExamples == other.positiveExamples &&
      negativeExamples == other.negativeExamples &&
      inputPrompt == other.inputPrompt &&
      fittedHeadBytes.contentEquals(other.fittedHeadBytes)
  }

  override fun hashCode(): Int {
    var result = condition.hashCode()
    result = 31 * result + threshold.hashCode()
    result = 31 * result + temperature.hashCode()
    result = 31 * result + normalizePrior.hashCode()
    result = 31 * result + positiveExamples.hashCode()
    result = 31 * result + negativeExamples.hashCode()
    result = 31 * result + inputPrompt.hashCode()
    result = 31 * result + (fittedHeadBytes?.contentHashCode() ?: 0)
    return result
  }

  override fun toString(): String =
    "BooleanQuestion(condition=$condition, threshold=$threshold, temperature=$temperature, " +
      "normalizePrior=$normalizePrior, positiveExamples=$positiveExamples, " +
      "negativeExamples=$negativeExamples, inputPrompt=$inputPrompt, " +
      "fittedHead=${fittedHeadBytes.describeBytes()})"

  /** Returns a copy of this question with the given fields replaced. */
  fun copy(
    condition: String = this.condition,
    threshold: Float = this.threshold,
    temperature: Float = this.temperature,
    normalizePrior: Boolean = this.normalizePrior,
    positiveExamples: List<String> = this.positiveExamples,
    negativeExamples: List<String> = this.negativeExamples,
    inputPrompt: InputPrompt = this.inputPrompt,
    fittedHead: ByteArray? = fittedHeadBytes,
  ): BooleanQuestion =
    BooleanQuestion(
      condition,
      threshold,
      temperature,
      normalizePrior,
      positiveExamples,
      negativeExamples,
      inputPrompt,
      fittedHead,
    )

  operator fun component1(): String = condition

  operator fun component2(): Float = threshold

  operator fun component3(): Float = temperature

  operator fun component4(): Boolean = normalizePrior

  operator fun component5(): List<String> = positiveExamples

  operator fun component6(): List<String> = negativeExamples

  operator fun component7(): InputPrompt = inputPrompt

  operator fun component8(): ByteArray? = fittedHead

  companion object {
    /** Creates a [BooleanQuestion] backed by a pre-fitted `FittedHead` byte blob. */
    @JvmStatic
    @JvmOverloads
    fun fromFittedHead(
      fittedHead: ByteArray,
      threshold: Float = 0.5f,
      temperature: Float = 0.0f,
      condition: String = "",
    ): BooleanQuestion =
      BooleanQuestion(
        condition = condition,
        threshold = threshold,
        temperature = temperature,
        fittedHead = fittedHead,
      )
  }
}

/** Describes an optional byte payload by size for `toString()`, without dumping its contents. */
private fun ByteArray?.describeBytes(): String = this?.let { "<${it.size} bytes>" } ?: "null"

/**
 * Evaluates a mutually exclusive categorical choice menu and returns a calibrated probability
 * distribution over [choices].
 *
 * @property choices Ordered list of option keys (may be empty when [fittedHead] is provided).
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
 * @param fittedHead Optional pre-fitted `FittedHead` binary payload for embedding models; copied on
 *   construction. Cannot be combined with [examples].
 */
// Not a data class so that the fittedHead payload can be defensively copied in and out.
// fittedHead is an opaque binary payload: ByteArray avoids boxing every byte in a List<Byte>.
@Suppress("KotlinArray")
class ChoiceQuestion
@JvmOverloads
constructor(
  val choices: List<String> = emptyList(),
  val descriptions: List<String> = emptyList(),
  val temperature: Float = 0.0f,
  val instructions: String = "",
  val scoringMode: ScoringMode = ScoringMode.LENGTH_NORMALIZED_LOG_PROB,
  val normalizePrior: Boolean = false,
  val examples: List<List<String>> = emptyList(),
  val inputPrompt: InputPrompt = InputPrompt.UNSPECIFIED,
  val balanceClassWeights: Boolean = true,
  fittedHead: ByteArray? = null,
) : DecisionQuestion {
  private val fittedHeadBytes: ByteArray? = fittedHead?.copyOf()

  /** A copy of the optional pre-fitted `FittedHead` binary payload for embedding models. */
  val fittedHead: ByteArray?
    get() = fittedHeadBytes?.copyOf()

  /** Whether any choice has an example. As in the C API, all-empty lists mean no examples. */
  internal val hasExamples: Boolean
    get() = examples.any { it.isNotEmpty() }

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
    fittedHead: ByteArray? = null,
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
    fittedHead = fittedHead,
  )

  init {
    require(choices.isNotEmpty() || fittedHeadBytes != null) {
      "ChoiceQuestion.choices must not be empty unless fittedHead is provided."
    }
    if (fittedHeadBytes != null) {
      require(fittedHeadBytes.isNotEmpty()) { "ChoiceQuestion.fittedHead must not be empty." }
      require(!hasExamples) { "A ChoiceQuestion cannot specify both fittedHead and examples." }
    }
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

  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is ChoiceQuestion) return false
    return choices == other.choices &&
      descriptions == other.descriptions &&
      temperature == other.temperature &&
      instructions == other.instructions &&
      scoringMode == other.scoringMode &&
      normalizePrior == other.normalizePrior &&
      examples == other.examples &&
      inputPrompt == other.inputPrompt &&
      balanceClassWeights == other.balanceClassWeights &&
      fittedHeadBytes.contentEquals(other.fittedHeadBytes)
  }

  override fun hashCode(): Int {
    var result = choices.hashCode()
    result = 31 * result + descriptions.hashCode()
    result = 31 * result + temperature.hashCode()
    result = 31 * result + instructions.hashCode()
    result = 31 * result + scoringMode.hashCode()
    result = 31 * result + normalizePrior.hashCode()
    result = 31 * result + examples.hashCode()
    result = 31 * result + inputPrompt.hashCode()
    result = 31 * result + balanceClassWeights.hashCode()
    result = 31 * result + (fittedHeadBytes?.contentHashCode() ?: 0)
    return result
  }

  override fun toString(): String =
    "ChoiceQuestion(choices=$choices, descriptions=$descriptions, temperature=$temperature, " +
      "instructions=$instructions, scoringMode=$scoringMode, normalizePrior=$normalizePrior, " +
      "examples=$examples, inputPrompt=$inputPrompt, balanceClassWeights=$balanceClassWeights, " +
      "fittedHead=${fittedHeadBytes.describeBytes()})"

  /** Returns a copy of this question with the given fields replaced. */
  fun copy(
    choices: List<String> = this.choices,
    descriptions: List<String> = this.descriptions,
    temperature: Float = this.temperature,
    instructions: String = this.instructions,
    scoringMode: ScoringMode = this.scoringMode,
    normalizePrior: Boolean = this.normalizePrior,
    examples: List<List<String>> = this.examples,
    inputPrompt: InputPrompt = this.inputPrompt,
    balanceClassWeights: Boolean = this.balanceClassWeights,
    fittedHead: ByteArray? = fittedHeadBytes,
  ): ChoiceQuestion =
    ChoiceQuestion(
      choices,
      descriptions,
      temperature,
      instructions,
      scoringMode,
      normalizePrior,
      examples,
      inputPrompt,
      balanceClassWeights,
      fittedHead,
    )

  operator fun component1(): List<String> = choices

  operator fun component2(): List<String> = descriptions

  operator fun component3(): Float = temperature

  operator fun component4(): String = instructions

  operator fun component5(): ScoringMode = scoringMode

  operator fun component6(): Boolean = normalizePrior

  operator fun component7(): List<List<String>> = examples

  operator fun component8(): InputPrompt = inputPrompt

  operator fun component9(): Boolean = balanceClassWeights

  operator fun component10(): ByteArray? = fittedHead

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
      fittedHead: ByteArray? = null,
    ): ChoiceQuestion =
      ChoiceQuestion(
        criteria = options,
        temperature = temperature,
        instructions = instructions,
        scoringMode = scoringMode,
        normalizePrior = normalizePrior,
        fittedHead = fittedHead,
      )

    /** Creates a [ChoiceQuestion] backed by a pre-fitted `FittedHead` byte blob. */
    @JvmStatic
    @JvmOverloads
    fun fromFittedHead(
      fittedHead: ByteArray,
      choices: List<String> = emptyList(),
      descriptions: List<String> = emptyList(),
      temperature: Float = 0.0f,
      instructions: String = "",
    ): ChoiceQuestion =
      ChoiceQuestion(
        choices = choices,
        descriptions = descriptions,
        temperature = temperature,
        instructions = instructions,
        fittedHead = fittedHead,
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
      inputPrompt: InputPrompt = InputPrompt.UNSPECIFIED,
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
