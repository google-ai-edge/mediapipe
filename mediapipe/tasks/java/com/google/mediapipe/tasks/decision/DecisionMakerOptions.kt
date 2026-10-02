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

import android.os.ParcelFileDescriptor
import java.nio.ByteBuffer

/** Hardware delegate selection for running MediaPipe Decision inference. */
enum class DecisionMakerDelegate(val value: Int) {
  CPU(0),
  GPU(1),
}

/**
 * Configuration options for initializing a [DecisionMaker] task instance.
 *
 * At least one of [modelPath], [modelAssetFileDescriptor], or [modelAssetBuffer] must be provided.
 */
data class DecisionMakerOptions(
  val modelPath: String = "",
  val modelAssetFileDescriptor: ParcelFileDescriptor? = null,
  val modelAssetBuffer: ByteBuffer? = null,
  val maxNumTokens: Int = 4096,
  val delegate: DecisionMakerDelegate = DecisionMakerDelegate.CPU,
) {
  /** Builder for [DecisionMakerOptions]. */
  class Builder {
    /** Filesystem path or Android asset path to the `.task`, `.tflite`, or `.litertlm` model. */
    var modelPath: String = ""
    /** Open [ParcelFileDescriptor] pointing to the model file. */
    var modelAssetFileDescriptor: ParcelFileDescriptor? = null
    /** Direct [ByteBuffer] containing the model bytes in memory. */
    var modelAssetBuffer: ByteBuffer? = null
    /** Maximum token sequence capacity (defaults to 4096). */
    var maxNumTokens: Int = 4096
    /** Hardware acceleration delegate (defaults to [DecisionMakerDelegate.CPU]). */
    var delegate: DecisionMakerDelegate = DecisionMakerDelegate.CPU

    /**
     * Sets the filesystem path or Android asset path to the `.task`, `.tflite`, or `.litertlm`
     * model.
     */
    fun setModelPath(modelPath: String): Builder {
      this.modelPath = modelPath
      return this
    }

    /** Sets an open [ParcelFileDescriptor] pointing to the model file. */
    fun setModelAssetFileDescriptor(fd: ParcelFileDescriptor?): Builder {
      this.modelAssetFileDescriptor = fd
      return this
    }

    /** Sets a direct [ByteBuffer] containing the model bytes in memory. */
    fun setModelAssetBuffer(buffer: ByteBuffer?): Builder {
      this.modelAssetBuffer = buffer
      return this
    }

    /** Sets the maximum token sequence capacity (defaults to 4096). */
    fun setMaxNumTokens(maxNumTokens: Int): Builder {
      this.maxNumTokens = maxNumTokens
      return this
    }

    /** Sets the hardware acceleration delegate (defaults to [DecisionMakerDelegate.CPU]). */
    fun setDelegate(delegate: DecisionMakerDelegate): Builder {
      this.delegate = delegate
      return this
    }

    /** Builds a validated [DecisionMakerOptions] instance. */
    fun build(): DecisionMakerOptions {
      require(maxNumTokens > 0) { "maxNumTokens must be positive, got $maxNumTokens." }
      val buffer = modelAssetBuffer
      if (buffer != null) {
        require(buffer.isDirect) { "modelAssetBuffer must be a direct ByteBuffer." }
      }
      return DecisionMakerOptions(
        modelPath = modelPath,
        modelAssetFileDescriptor = modelAssetFileDescriptor,
        modelAssetBuffer = buffer,
        maxNumTokens = maxNumTokens,
        delegate = delegate,
      )
    }
  }

  companion object {
    /** Creates a new [Builder] for [DecisionMakerOptions]. */
    @JvmStatic fun builder(): Builder = Builder()

    /** Kotlin DSL helper for constructing [DecisionMakerOptions]. */
    inline fun decisionMakerOptions(block: Builder.() -> Unit): DecisionMakerOptions =
      builder().apply(block).build()
  }
}
