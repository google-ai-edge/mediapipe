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
import android.util.Log
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.IntentSenderRequest
import com.google.android.play.core.aipacks.AiPackLocation
import com.google.android.play.core.aipacks.AiPackManager
import com.google.android.play.core.aipacks.AiPackManagerFactory
import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.model.AiPackErrorCode
import com.google.android.play.core.aipacks.model.AiPackStatus
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
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
open class VisionProviderBase : AutoCloseable {
  private val context: Context
  private val aiPackManager: AiPackManager
  private val executor = Executors.newCachedThreadPool()
  private val aiPackDownloadListeners =
    mutableMapOf<VisionModel, MutableList<AiPackDownloadListener>>()

  constructor(context: Context) : this(context, AiPackManagerFactory.getInstance(context))

  constructor(context: Context, aiPackManager: AiPackManager) {
    this.context = context
    this.aiPackManager = aiPackManager
  }

  /**
   * Adds a listener for model download events.
   *
   * @param model The vision model to listen for.
   * @param listener The listener to add.
   */
  fun addModelDownloadListener(model: VisionModel, listener: AiPackDownloadListener) {
    synchronized(aiPackDownloadListeners) {
      val listeners = aiPackDownloadListeners.getOrPut(model) { mutableListOf() }
      listeners.add(listener)
    }
  }

  /**
   * Removes a previously added model download listener.
   *
   * @param model The vision model the listener is registered to.
   * @param listener The listener to remove.
   */
  fun removeModelDownloadListener(model: VisionModel, listener: AiPackDownloadListener) {
    synchronized(aiPackDownloadListeners) {
      aiPackDownloadListeners[model]?.let {
        it.remove(listener)
        if (it.isEmpty()) {
          aiPackDownloadListeners.remove(model)
        }
      }
    }
  }

  /**
   * Requests a download over cellular data.
   *
   * @param launcher The ActivityResultLauncher registered in your Activity/Fragment to handle
   *   StartIntentSenderForResult.
   */
  fun requestCellularDataDownload(launcher: ActivityResultLauncher<IntentSenderRequest>) {
    // Show the confirmation dialog to the user. The result will be handled by the
    // download listeners.
    val unused = aiPackManager.showConfirmationDialog(launcher)
  }

  /** Helper function to notify all registered listeners for a specific model. */
  private fun notifyModelDownloadListeners(
    model: VisionModel,
    state: AiPackState,
    exception: Exception? = null,
  ) {
    synchronized(aiPackDownloadListeners) {
      aiPackDownloadListeners[model]?.forEach { it.updateState(state, exception) }
    }
  }

  /**
   * Constructs the absolute path for an AI Ai within a given [AiPackLocation].
   *
   * @param packLocation The [AiPackLocation] of the installed AI Pack.
   * @param relativeAiAiPath The relative path of the Ai within the AI Pack.
   * @return The absolute file system path to the AI Ai.
   */
  private fun getAbsoluteAiAssetPath(
    packLocation: AiPackLocation,
    relativeAiAssetPath: String,
  ): String {
    return "${packLocation.assetsPath()}/$relativeAiAssetPath"
  }

  /**
   * Unified method to load the model file into a ByteBuffer. It prioritizes the downloaded AI Pack
   * and falls back to the app's local Ais.
   *
   * @param model The [VisionModel] for which to load the buffer.
   * @return A [ByteBuffer] containing the model data.
   * @throws FileNotFoundException if the model file cannot be found in either AI Packs or Ais.
   * @throws IOException if there's an error reading the file.
   */
  private fun getModelAsBuffer(model: VisionModel): ByteBuffer {
    val packName = "aipack_" + model.enumName
    val filename = model.createModelFileName()
    val packLocation = aiPackManager.getPackLocation(packName)

    // Prioritize downloaded AI Pack if available
    if (packLocation?.assetsPath() != null) {
      val modelPath = getAbsoluteAiAssetPath(packLocation, filename)
      val modelFile = File(modelPath)
      if (modelFile.exists()) {
        return FileInputStream(modelFile).use { inputStream ->
          inputStream.channel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length())
        }
      } else {
        throw FileNotFoundException("Model file not found: " + modelFile)
      }
    }

    // Fallback to model included in the app's directory
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
    val packName = "aipack_" + model.enumName

    val future = CompletableFuture<T>()
    try {
      if (aiPackManager.getPackLocation(packName) != null) {
        Log.d("VisionProvider", "AI Pack '$packName' already installed. Creating task.")
        instantiateTaskWithModel(future, packName, model, settings, creator)
      } else {
        Log.d("VisionProvider", "AI Pack '$packName' not found. Starting download.")
        initiateAiPackDownloadForTask(future, packName, model, settings, creator)
      }
    } catch (e: Exception) {
      future.completeExceptionally(e)
    }
    return future
  }

  /**
   * Initiates the download process for a given AI Pack and creates the task on completion.
   *
   * @param future The [CompletableFuture] to complete with the created task instance.
   * @param packName The name of the AI Pack to download.
   * @param model The [VisionModel] associated with the AI Pack.
   * @param settings The task-specific internal settings.
   * @param creator A lambda function that creates the task instance.
   */
  private fun <T, S> initiateAiPackDownloadForTask(
    future: CompletableFuture<T>,
    packName: String,
    model: VisionModel,
    settings: S,
    creator: (Context, ByteBuffer, S) -> T,
  ) {
    val downloader =
      AiPackDownloader(
        context,
        executor,
        aiPackManager,
        packName,
        object : AiPackDownloadListener() {
          override fun onProgress(progress: Float, state: AiPackState) {
            notifyModelDownloadListeners(model, state)
          }

          override fun onCompleted() {
            instantiateTaskWithModel(future, packName, model, settings, creator)
          }

          override fun onFailed(e: Exception, state: AiPackState) {
            notifyModelDownloadListeners(model, state, exception = e)
            future.completeExceptionally(e)
          }

          override fun onWaitingForWifi(state: AiPackState) {
            notifyModelDownloadListeners(model, state)
          }

          override fun onShowConfirmationDialog(state: AiPackState) {
            notifyModelDownloadListeners(model, state)
          }
        },
      )
    downloader.downloadPack()
  }

  private fun <T, S> instantiateTaskWithModel(
    future: CompletableFuture<T>,
    packName: String,
    model: VisionModel,
    settings: S,
    creator: (Context, ByteBuffer, S) -> T,
  ) {
    val unused =
      executor.submit {
        try {
          val modelBuffer = getModelAsBuffer(model)
          notifyModelDownloadListeners(
            model,
            state =
              AiPackState.create(
                packName,
                AiPackStatus.COMPLETED,
                AiPackErrorCode.NO_ERROR,
                0,
                modelBuffer.remaining().toLong(),
                1.0,
              ),
          )
          val task = creator(context, modelBuffer, settings)
          future.complete(task)
        } catch (e: Exception) {
          future.completeExceptionally(e)
        }
      }
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
