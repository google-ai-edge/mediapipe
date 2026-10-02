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

import android.content.Context
import android.os.ParcelFileDescriptor
import android.util.Log
import com.google.mediapipe.tasks.core.BaseOptionsUtils
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.atomic.AtomicBoolean

/**
 * MediaPipe Decision Task API for Android / Kotlin.
 *
 * Evaluates structured decision questions ([BooleanQuestion], [ChoiceQuestion], [ScoreQuestion],
 * and multi-question JSON schemas) against text or multimodal [DecisionContext] inputs in a single
 * forward pass, returning calibrated probabilities, expected rubric scores, and confidence metrics.
 */
class DecisionMaker private constructor(private var handle: Long) : Closeable, AutoCloseable {

  private val closed = AtomicBoolean(false)

  /** Evaluates a [BooleanQuestion] against [text] and returns a calibrated [BooleanResult]. */
  fun evaluate(text: String, question: BooleanQuestion): BooleanResult {
    checkNotClosed()
    return nativeEvaluateBoolean(
      handle,
      text,
      question.condition,
      question.threshold,
      question.temperature,
      question.normalizePrior,
    )
  }

  /** Evaluates a [ChoiceQuestion] against [text] and returns a calibrated [ChoiceResult]. */
  fun evaluate(text: String, question: ChoiceQuestion): ChoiceResult {
    checkNotClosed()
    return nativeEvaluateChoice(
      handle,
      text,
      question.choices.toTypedArray(),
      if (question.descriptions.isEmpty()) null else question.descriptions.toTypedArray(),
      question.temperature,
      question.instructions,
      question.scoringMode.value,
      question.normalizePrior,
    )
  }

  /** Evaluates a [ScoreQuestion] against [text] and returns a calibrated [ScoreResult]. */
  fun evaluate(text: String, question: ScoreQuestion): ScoreResult {
    checkNotClosed()
    return nativeEvaluateScore(
      handle,
      text,
      question.rubric.toTypedArray(),
      question.temperature,
      question.instructions,
    )
  }

  /**
   * Evaluates a [BooleanQuestion] against a batch of [texts] (with an optional [sharedPrefixText]
   * for shared-prefix candidate batching).
   */
  fun evaluateBatch(
    texts: List<String>,
    question: BooleanQuestion,
    sharedPrefixText: String? = null,
  ): List<BooleanResult> {
    checkNotClosed()
    if (texts.isEmpty()) return emptyList()
    return nativeEvaluateBooleanBatch(
        handle,
        sharedPrefixText,
        texts.toTypedArray(),
        question.condition,
        question.threshold,
        question.temperature,
        question.normalizePrior,
      )
      .toList()
  }

  /**
   * Evaluates a [ChoiceQuestion] against a batch of [texts] (with an optional [sharedPrefixText]
   * for shared-prefix candidate batching).
   */
  fun evaluateBatch(
    texts: List<String>,
    question: ChoiceQuestion,
    sharedPrefixText: String? = null,
  ): List<ChoiceResult> {
    checkNotClosed()
    if (texts.isEmpty()) return emptyList()
    return nativeEvaluateChoiceBatch(
        handle,
        sharedPrefixText,
        texts.toTypedArray(),
        question.choices.toTypedArray(),
        if (question.descriptions.isEmpty()) null else question.descriptions.toTypedArray(),
        question.temperature,
        question.instructions,
        question.scoringMode.value,
        question.normalizePrior,
      )
      .toList()
  }

  /**
   * Evaluates a [ScoreQuestion] against a batch of [texts] (with an optional [sharedPrefixText] for
   * shared-prefix candidate batching).
   */
  fun evaluateBatch(
    texts: List<String>,
    question: ScoreQuestion,
    sharedPrefixText: String? = null,
  ): List<ScoreResult> {
    checkNotClosed()
    if (texts.isEmpty()) return emptyList()
    return nativeEvaluateScoreBatch(
        handle,
        sharedPrefixText,
        texts.toTypedArray(),
        question.rubric.toTypedArray(),
        question.temperature,
        question.instructions,
      )
      .toList()
  }

  /**
   * Evaluates a JSON request containing input text (`text`, `context`, or `state`) and a
   * multi-question schema (`questions` or OpenAI `properties`) and returns a JSON-serialized
   * `DecisionResult`.
   */
  fun evaluateJson(jsonRequest: String): String {
    checkNotClosed()
    return nativeEvaluateJson(handle, jsonRequest)
  }

  /**
   * Evaluates a JSON question schema against a multimodal or text [DecisionContext] and returns a
   * JSON-serialized `DecisionResult`.
   */
  fun evaluateJson(context: DecisionContext, jsonQuestions: String): String {
    checkNotClosed()
    return nativeEvaluateContextJson(
      handle,
      context.text,
      context.imageRgb,
      context.imageWidth,
      context.imageHeight,
      context.imageChannels,
      context.imageBytes,
      context.audioSamples,
      context.audioSampleRate,
      jsonQuestions,
    )
  }

  /** Prewarms the prompt prefix and option KV/embedding cache for a [ChoiceQuestion]. */
  fun prewarm(question: ChoiceQuestion) {
    checkNotClosed()
    nativePrewarmChoice(
      handle,
      question.choices.toTypedArray(),
      if (question.descriptions.isEmpty()) null else question.descriptions.toTypedArray(),
      question.temperature,
      question.instructions,
      question.scoringMode.value,
      question.normalizePrior,
    )
  }

  /** Prewarms and caches a JSON question schema into the backend's KV/option cache. */
  fun prewarmJson(jsonSchema: String) {
    checkNotClosed()
    nativePrewarmJson(handle, jsonSchema)
  }

  override fun close() {
    if (closed.compareAndSet(false, true)) {
      val currentHandle = handle
      handle = 0L
      if (currentHandle != 0L) {
        nativeClose(currentHandle)
      }
    }
  }

  private fun checkNotClosed() {
    check(!closed.get() && handle != 0L) { "DecisionMaker instance has already been closed." }
  }

  private external fun nativeEvaluateBoolean(
    handle: Long,
    text: String,
    condition: String,
    threshold: Float,
    temperature: Float,
    normalizePrior: Boolean,
  ): BooleanResult

  private external fun nativeEvaluateChoice(
    handle: Long,
    text: String,
    keys: Array<String>,
    descriptions: Array<String>?,
    temperature: Float,
    instructions: String,
    scoringMode: Int,
    normalizePrior: Boolean,
  ): ChoiceResult

  private external fun nativeEvaluateScore(
    handle: Long,
    text: String,
    rubric: Array<String>,
    temperature: Float,
    instructions: String,
  ): ScoreResult

  private external fun nativeEvaluateBooleanBatch(
    handle: Long,
    sharedPrefixText: String?,
    texts: Array<String>,
    condition: String,
    threshold: Float,
    temperature: Float,
    normalizePrior: Boolean,
  ): Array<BooleanResult>

  private external fun nativeEvaluateChoiceBatch(
    handle: Long,
    sharedPrefixText: String?,
    texts: Array<String>,
    keys: Array<String>,
    descriptions: Array<String>?,
    temperature: Float,
    instructions: String,
    scoringMode: Int,
    normalizePrior: Boolean,
  ): Array<ChoiceResult>

  private external fun nativeEvaluateScoreBatch(
    handle: Long,
    sharedPrefixText: String?,
    texts: Array<String>,
    rubric: Array<String>,
    temperature: Float,
    instructions: String,
  ): Array<ScoreResult>

  private external fun nativeEvaluateJson(handle: Long, jsonRequest: String): String

  private external fun nativeEvaluateContextJson(
    handle: Long,
    text: String,
    imageRgb: ByteArray?,
    imageWidth: Int,
    imageHeight: Int,
    imageChannels: Int,
    imageBytes: ByteArray?,
    audioSamples: FloatArray?,
    audioSampleRate: Int,
    jsonQuestions: String,
  ): String

  private external fun nativePrewarmChoice(
    handle: Long,
    keys: Array<String>,
    descriptions: Array<String>?,
    temperature: Float,
    instructions: String,
    scoringMode: Int,
    normalizePrior: Boolean,
  )

  private external fun nativePrewarmJson(handle: Long, jsonSchema: String)

  private external fun nativeClose(handle: Long)

  companion object {
    private const val TAG = "MediaPipeDecision"

    init {
      try {
        System.loadLibrary("mediapipe_tasks_decision_jni")
      } catch (e: UnsatisfiedLinkError) {
        Log.w(TAG, "Failed to load mediapipe_tasks_decision_jni: ${e.message}")
      }
    }

    /**
     * Creates a [DecisionMaker] instance from a model asset path or filesystem path.
     *
     * If [modelPath] refers to a bundled asset in `context.assets` rather than an existing
     * filesystem path, it is automatically staged to `context.cacheDir` so native memory-mapping
     * can open it directly.
     */
    @JvmStatic
    fun createFromFile(context: Context, modelPath: String): DecisionMaker {
      return createFromOptions(
        context,
        DecisionMakerOptions.builder().setModelPath(modelPath).build(),
      )
    }

    /** Creates a [DecisionMaker] instance from [DecisionMakerOptions]. */
    @JvmStatic
    fun createFromOptions(context: Context, options: DecisionMakerOptions): DecisionMaker {
      val appVersion = BaseOptionsUtils.getAppVersion(context)
      val appId = BaseOptionsUtils.getAppId(context)

      var fd = -1
      val pfd: ParcelFileDescriptor? = options.modelAssetFileDescriptor
      if (pfd != null) {
        try {
          fd = pfd.dup().detachFd()
        } catch (e: IOException) {
          if (options.modelPath.isEmpty() && options.modelAssetBuffer == null) {
            throw IllegalArgumentException(
              "Decision: Failed to read from file descriptor and model path/buffer are empty.",
              e,
            )
          }
          Log.w(TAG, "Failed to dup file descriptor, falling back to model path/buffer.", e)
        }
      }

      var resolvedPath = options.modelPath
      if (resolvedPath.isNotEmpty() && !File(resolvedPath).exists()) {
        // Check if the path exists inside Android assets; if so, copy to cacheDir.
        try {
          context.assets.open(resolvedPath).use { inputStream ->
            val cachedFile = File(context.cacheDir, File(resolvedPath).name)
            FileOutputStream(cachedFile).use { outputStream -> inputStream.copyTo(outputStream) }
            resolvedPath = cachedFile.absolutePath
          }
        } catch (_: IOException) {
          // Keep original resolvedPath so the native layer reports a clear error message.
        }
      }

      val handle =
        nativeCreateHandle(
          fd,
          resolvedPath,
          options.modelAssetBuffer,
          options.maxNumTokens,
          options.delegate.value,
          appId,
          appVersion,
          BaseOptionsUtils.HOST_ENVIRONMENT_ANDROID,
          BaseOptionsUtils.HOST_SYSTEM_ANDROID,
        )
      return DecisionMaker(handle)
    }

    @JvmStatic
    private external fun nativeCreateHandle(
      modelFd: Int,
      modelPath: String,
      modelBuffer: ByteBuffer?,
      maxNumTokens: Int,
      delegate: Int,
      appId: String,
      appVersion: String,
      hostEnvironment: Int,
      hostSystem: Int,
    ): Long
  }
}
