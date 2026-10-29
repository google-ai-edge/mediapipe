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

@file:Suppress("PackageName", "KotlinArray")

package com.google.mediapipe.tasks.audio.texttospeech

import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.BaseOptionsUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext

/** Voice metadata descriptor returned by [TextToSpeech.listVoices]. */
data class VoiceInfo(
  val voiceId: String,
  val displayName: String,
  val locale: String,
  val sampleRateHz: Int,
  val isDownloaded: Boolean,
)

/** Word-level timing alignment mapping an input word to audio seconds. */
data class WordTimestamp(
  val word: String,
  val startChar: Int,
  val endChar: Int,
  val startSec: Float,
  val endSec: Float,
)

/** Self-contained mono PCM audio buffer returned by [TextToSpeech.synthesize]. */
data class AudioResult
@JvmOverloads
constructor(
  val samples: FloatArray,
  val sampleRateHz: Int = 24000,
  val wordTimestamps: List<WordTimestamp> = emptyList(),
) {
  val durationSeconds: Float
    get() = if (sampleRateHz > 0) samples.size.toFloat() / sampleRateHz.toFloat() else 0f

  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is AudioResult) return false
    return samples.contentEquals(other.samples) &&
      sampleRateHz == other.sampleRateHz &&
      wordTimestamps == other.wordTimestamps
  }

  override fun hashCode(): Int {
    var result = samples.contentHashCode()
    result = 31 * result + sampleRateHz
    result = 31 * result + wordTimestamps.hashCode()
    return result
  }
}

typealias TtsAudioResult = AudioResult

/** Incremental mono PCM audio chunk emitted by [TextToSpeech.synthesizeStream]. */
data class AudioChunk
@JvmOverloads
constructor(
  val samples: FloatArray,
  val sampleRateHz: Int = 24000,
  val chunkIndex: Int = 0,
  val isFinal: Boolean = false,
  val wordTimestamps: List<WordTimestamp> = emptyList(),
) {
  val isFinalChunk: Boolean
    get() = isFinal

  val durationSeconds: Float
    get() = if (sampleRateHz > 0) samples.size.toFloat() / sampleRateHz.toFloat() else 0f

  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (other !is AudioChunk) return false
    return samples.contentEquals(other.samples) &&
      sampleRateHz == other.sampleRateHz &&
      chunkIndex == other.chunkIndex &&
      isFinal == other.isFinal &&
      wordTimestamps == other.wordTimestamps
  }

  override fun hashCode(): Int {
    var result = samples.contentHashCode()
    result = 31 * result + sampleRateHz
    result = 31 * result + chunkIndex
    result = 31 * result + isFinal.hashCode()
    result = 31 * result + wordTimestamps.hashCode()
    return result
  }
}

/**
 * Bidirectional / pull-based streaming synthesis handle supporting Level 2 incremental LLM text
 * pushing ([pushText] + [finishInput]) alongside [readNext] and [cancel].
 */
class TtsStream internal constructor(@Volatile private var streamHandle: Long) : AutoCloseable {
  private val lock = Any()

  /** Appends an incremental text fragment into the sentence chunker. */
  fun pushText(textDelta: String) {
    val current = streamHandle
    check(current != 0L) { "TtsStream is closed." }
    MediaPipeTtsJni.nativeStreamPushText(current, textDelta)
  }

  /** Signals that no more text fragments will be pushed, flushing any trailing partial sentence. */
  fun finishInput() {
    val current = streamHandle
    check(current != 0L) { "TtsStream is closed." }
    MediaPipeTtsJni.nativeStreamFinishInput(current)
  }

  /**
   * Pulls the next synthesized [AudioChunk] synchronously on the calling thread, or returns `null`
   * when the stream reaches end-of-stream.
   */
  fun readNext(): AudioChunk? {
    val current = streamHandle
    check(current != 0L) { "TtsStream is closed." }
    return MediaPipeTtsJni.nativeStreamReadNext(current)
  }

  /** Lock-free cancellation for this stream. */
  fun cancel() {
    val current = streamHandle
    if (current != 0L) {
      MediaPipeTtsJni.nativeStreamCancel(current)
    }
  }

  override fun close() {
    cancel()
    synchronized(lock) {
      val current = streamHandle
      if (current != 0L) {
        streamHandle = 0L
        MediaPipeTtsJni.nativeStreamDestroy(current)
      }
    }
  }
}

/** Task initialization options for [TextToSpeech] (backed by LiteRT-LM Omni Kokoro-82M). */
data class TextToSpeechOptions(
  val baseOptions: BaseOptions,
  val defaultVoiceId: String? = null,
  val defaultLocale: String? = null,
  val speakingRate: Float = 1.0f,
  val targetBucket: Int = 0,
  val cacheDirectory: String = "",
)

/**
 * Per-utterance synthesis overrides passed to [TextToSpeech.synthesize] and
 * [TextToSpeech.synthesizeStream].
 *
 * Both fields default to `null` so non-English voices (e.g. `"ef_dora"`, `"zf_xiaobei"`)
 * automatically infer their native language from the voice metadata.
 */
data class SynthesisOptions(val voiceId: String? = null, val locale: String? = null)

/** MediaPipe TextToSpeech task interface for Kotlin / Android (Unary + Streaming API). */
class TextToSpeech private constructor(@Volatile private var nativeHandle: Long) : AutoCloseable {

  companion object {
    /** Internal sentinel path used for fast, hermetic unit testing without a `.litertlm` file. */
    const val SYNTHETIC_TEST_MODEL_PATH = ":synthetic_test_model:"

    internal fun toNativeDelegate(baseOptions: BaseOptions): Int {
      val acceleration = BaseOptionsUtils.convertBaseOptionsToProto(baseOptions).acceleration
      return when {
        acceleration.hasGpu() -> 1
        acceleration.hasLitert() && acceleration.litert.hasNpu() -> 4
        else -> 0
      }
    }

    /** Creates a [TextToSpeech] task instance configured with [options]. */
    @JvmStatic
    fun create(options: TextToSpeechOptions): TextToSpeech {
      val baseOptionsProto = BaseOptionsUtils.convertBaseOptionsToProto(options.baseOptions)
      val handle =
        MediaPipeTtsJni.nativeCreate(
          modelAssetPath = baseOptionsProto.modelAsset.fileName,
          delegate = toNativeDelegate(options.baseOptions),
          speakingRate = options.speakingRate,
          targetBucket = options.targetBucket,
          defaultVoiceId = options.defaultVoiceId,
          defaultLocale = options.defaultLocale,
          cacheDirectory = options.cacheDirectory,
        )
      return TextToSpeech(handle)
    }

    /**
     * Computes model-agnostic word-level timestamps (`startSec` / `endSec` and character offsets)
     * for [text] aligned against mono PCM [samples] at [sampleRateHz].
     */
    @JvmStatic
    @JvmOverloads
    fun computeWordTimestamps(
      text: String,
      samples: FloatArray,
      sampleRateHz: Int = 24000,
      timeOffsetSec: Float = 0f,
    ): List<WordTimestamp> {
      if (text.isEmpty() || samples.isEmpty() || sampleRateHz <= 0) return emptyList()
      return MediaPipeTtsJni.nativeComputeWordTimestamps(
        text = text,
        samples = samples,
        sampleRateHz = sampleRateHz,
        timeOffsetSec = timeOffsetSec,
      )
    }
  }

  private val synthLock = Any()

  /**
   * Queries available voices for this initialized [TextToSpeech] instance and its `.litertlm`
   * bundle.
   */
  fun listVoices(): List<VoiceInfo> {
    check(nativeHandle != 0L) { "TextToSpeech is closed." }
    return MediaPipeTtsJni.nativeListVoices(nativeHandle).toList()
  }

  /** Synchronously synthesizes [text] on the calling thread and returns the [AudioResult]. */
  @JvmOverloads
  fun synthesize(text: String, options: SynthesisOptions? = null): AudioResult {
    synchronized(synthLock) {
      check(nativeHandle != 0L) { "TextToSpeech is closed." }
      return MediaPipeTtsJni.nativeSynthesize(
        handle = nativeHandle,
        text = text,
        voiceId = options?.voiceId,
        locale = options?.locale,
      )
    }
  }

  /**
   * Streams synthesized PCM [AudioChunk]s as each sentence or vocoder slice completes.
   *
   * Cancelling the collecting coroutine (`take(1)`, `first()`, or `job.cancel()`) unconditionally
   * executes `nativeStreamCancel` + `nativeStreamDestroy` in `finally`, immediately halting
   * in-flight native inference.
   */
  @JvmOverloads
  fun synthesizeStream(text: String, options: SynthesisOptions? = null): Flow<AudioChunk> = flow {
    val current = nativeHandle
    check(current != 0L) { "TextToSpeech is closed." }
    val streamHandle =
      MediaPipeTtsJni.nativeStartStream(
        handle = current,
        initialText = text,
        voiceId = options?.voiceId,
        locale = options?.locale,
        finishInput = true,
      )
    val completionHandle =
      currentCoroutineContext()[Job]?.invokeOnCompletion { cause ->
        if (cause != null) {
          MediaPipeTtsJni.nativeStreamCancel(streamHandle)
        }
      }
    try {
      while (true) {
        currentCoroutineContext().ensureActive()
        val chunk = MediaPipeTtsJni.nativeStreamReadNext(streamHandle) ?: break
        emit(chunk)
      }
    } finally {
      completionHandle?.dispose()
      MediaPipeTtsJni.nativeStreamCancel(streamHandle)
      MediaPipeTtsJni.nativeStreamDestroy(streamHandle)
    }
  }

  /**
   * Opens a [TtsStream] for Level 2 incremental LLM text streaming
   * ([TtsStream.pushText] + [TtsStream.finishInput]) or manual pull iteration
   * ([TtsStream.readNext]).
   */
  @JvmOverloads
  fun openStream(
    initialText: String = "",
    options: SynthesisOptions? = null,
    finishInput: Boolean = false,
  ): TtsStream {
    val current = nativeHandle
    check(current != 0L) { "TextToSpeech is closed." }
    val streamHandle =
      MediaPipeTtsJni.nativeStartStream(
        handle = current,
        initialText = initialText,
        voiceId = options?.voiceId,
        locale = options?.locale,
        finishInput = finishInput,
      )
    return TtsStream(streamHandle)
  }

  /** Opens an incremental [TtsStream] configured with [options] and empty initial text. */
  fun openStream(options: SynthesisOptions): TtsStream =
    openStream(initialText = "", options = options, finishInput = false)

  /**
   * Cancellable suspending helper that executes [synthesize] off the calling thread and
   * automatically triggers [cancel] if the calling coroutine is cancelled.
   */
  suspend fun synthesizeAsync(text: String, options: SynthesisOptions? = null): AudioResult =
    withContext(Dispatchers.Default) {
      val completionHandle =
        currentCoroutineContext()[Job]?.invokeOnCompletion { cause ->
          if (cause != null) {
            cancel()
          }
        }
      try {
        synthesize(text, options)
      } finally {
        completionHandle?.dispose()
      }
    }

  /**
   * Thread-safe, lock-free barge-in cancellation. Immediately aborts any in-flight [synthesize] or
   * [synthesizeStream] call.
   */
  fun cancel() {
    val current = nativeHandle
    if (current != 0L) {
      MediaPipeTtsJni.nativeCancel(current)
    }
  }

  override fun close() {
    cancel()
    synchronized(synthLock) {
      val current = nativeHandle
      if (current != 0L) {
        nativeHandle = 0L
        MediaPipeTtsJni.nativeDestroy(current)
      }
    }
  }
}
