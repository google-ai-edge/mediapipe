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

import android.content.Context
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executors
import java.util.concurrent.Future

/**
 * Base class for VisionProviders, handling the common logic for downloading and managing MediaPipe
 * Tasks models and NPU delegate libraries.
 *
 * @property context The application [Context].
 */
open class VisionProviderBase constructor(private val context: Context) : AutoCloseable {
  private val executor = Executors.newCachedThreadPool()

  /**
   * Unified method to load the model file into a ByteBuffer.
   *
   * @param model The [VisionModel] for which to load the buffer.
   * @return A [ByteBuffer] containing the model data.
   * @throws FileNotFoundException if the model file cannot be found in either AI Packs or Ais.
   * @throws IOException if there's an error reading the file.
   */
  private fun getModelAsBuffer(model: VisionModel): ByteBuffer {
    val filename = model.createModelFileName()
    return context.assets.openFd("model/$filename").use { afd ->
      FileInputStream(afd.fileDescriptor).use { inputStream ->
        inputStream.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
      }
    }
  }

  /**
   * Generic helper to create a BaseOptions builder with its model buffer.
   *
   * @param modelAiBuffer The [ByteBuffer] containing the model.
   * @return A [BaseOptions.Builder] instance.
   */
  private fun createBaseOptions(modelAssetBuffer: ByteBuffer): BaseOptions.Builder {
    return BaseOptions.builder().setModelAssetBuffer(modelAssetBuffer)
  }

  /**
   * Generic function to create any MediaPipe Vision task.
   *
   * @param model The [VisionModel] to be used for the task.
   * @param settings The task-specific internal settings.
   * @param creator A lambda function that creates the task instance.
   * @return A [Future] that will complete with the created task instance.
   */
  private fun <T, S> createTask(
    model: VisionModel,
    settings: S,
    creator: (Context, ByteBuffer, S) -> T,
  ): Future<T> {
    val future = CompletableFuture<T>()
    try {
      val unused =
        executor.submit {
          try {
            val modelBuffer = getModelAsBuffer(model)
            val task = creator(context, modelBuffer, settings)
            future.complete(task)
          } catch (e: Exception) {
            future.completeExceptionally(e)
          }
        }
    } catch (e: Exception) {
      future.completeExceptionally(e)
    }
    return future
  }

  /**
   * Creates an [ImageSegmenter] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containing the model.
   * @param settings The internal settings for the [ImageSegmenter].
   * @return An [ImageSegmenter] instance.
   */
  private fun createImageSegmenterImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    settings: ImageSegmenterSettingsInternal,
  ): ImageSegmenter {
    val baseOptions = createBaseOptions(modelBuffer).build()
    val options = settings.toOptions(baseOptions)
    return ImageSegmenter.createFromOptions(context, options)
  }

  /**
   * Creates an [ImageSegmenter] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [ImageSegmenter].
   * @return A [Future] that completes with the created [ImageSegmenter].
   */
  fun createImageSegmenterImpl(
    model: VisionModel,
    settings: ImageSegmenterSettingsInternal,
  ): Future<ImageSegmenter> = createTask(model, settings, ::createImageSegmenterImplHelper)

  /** Closes the VisionProvider and releases all resources. */
  override fun close() {
    executor.shutdown()
  }
}
