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
import com.google.android.play.core.aipacks.AiPackState
import com.google.android.play.core.aipacks.model.AiPackErrorCode
import com.google.android.play.core.aipacks.model.AiPackStatus
import com.google.android.play.core.splitinstall.SplitInstallManager
import com.google.android.play.core.splitinstall.SplitInstallRequest
import com.google.android.play.core.splitinstall.SplitInstallSessionState
import com.google.android.play.core.splitinstall.SplitInstallStateUpdatedListener
import com.google.android.play.core.splitinstall.model.SplitInstallErrorCode
import com.google.android.play.core.splitinstall.model.SplitInstallSessionStatus
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.gesturerecognizer.GestureRecognizer
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.imageclassifier.ImageClassifier
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
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
open class VisionProviderBase
constructor(
  private val context: Context,
  private val aiPackManager: AiPackManager,
  private val splitInstallManager: SplitInstallManager,
) : AutoCloseable {
  private val executor = Executors.newCachedThreadPool()
  private val aiPackDownloadListeners =
    mutableMapOf<VisionModel, MutableList<AiPackDownloadListener>>()
  private val npuDelegateDownloadListeners = mutableListOf<NpuDelegateDownloadListener>()
  private val npuDelegateLibraryPath by lazy { downloadNpuDelegateInternal() }

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
   * Adds a listener for NPU delegate download events.
   *
   * @param listener The listener to add.
   */
  fun addNpuDelegateDownloadListener(listener: NpuDelegateDownloadListener) {
    synchronized(npuDelegateDownloadListeners) { npuDelegateDownloadListeners.add(listener) }
  }

  /**
   * Removes a previously added NPU delegate download listener.
   *
   * @param listener The listener to remove.
   */
  fun removeNpuDelegateDownloadListener(listener: NpuDelegateDownloadListener) {
    synchronized(npuDelegateDownloadListeners) { npuDelegateDownloadListeners.remove(listener) }
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

  /** Helper function to notify all registered listeners for NPU delegate downloads. */
  private fun notifyNpuListeners(action: (NpuDelegateDownloadListener) -> Unit) {
    synchronized(npuDelegateDownloadListeners) { npuDelegateDownloadListeners.forEach(action) }
  }

  /**
   * Downloads the first available NPU Play Feature Delivery module from a prioritized list.
   *
   * This function iterates through a list of candidate modules (e.g., different NPU library
   * versions) and attempts to install them one by one. It stops as soon as one is successfully
   * installed or gracefully fails if none are compatible with the device.
   *
   * @param context The application context.
   * @return A CompletableFuture that completes with the native library path if a module is
   *   installed, or null if no compatible module is found or an error occurs.
   */
  private fun downloadNpuDelegateInternal(): CompletableFuture<String?> {
    val future = CompletableFuture<String?>()

    // Check if any of the candidate modules are already installed.
    val installedModule =
      NPU_DELEGATE_CANDIDATES.firstOrNull { splitInstallManager.installedModules.contains(it) }
    if (installedModule != null) {
      Log.d(TAG, "NPU delegate module $installedModule already installed")
      future.complete(context.applicationInfo.nativeLibraryDir)
      return future
    }
    Log.d(TAG, "NPU delegate module not installed, attempting download")

    // If not installed, request all modules. Only modules that are available for device and not yet
    // installed will be downloaded.
    val listener =
      object : SplitInstallStateUpdatedListener {
        override fun onStateUpdate(state: SplitInstallSessionState) {
          notifyNpuListeners { it.updateState(state) }
          when (state.status()) {
            SplitInstallSessionStatus.INSTALLED -> {
              splitInstallManager.unregisterListener(this)
              future.complete(context.applicationInfo.nativeLibraryDir)
            }
            SplitInstallSessionStatus.FAILED,
            SplitInstallSessionStatus.CANCELED -> {
              splitInstallManager.unregisterListener(this)
              future.complete(null)
            }
            else -> {
              // Other states are handled by NpuDelegateDownloadListener.
            }
          }
        }
      }
    splitInstallManager.registerListener(listener)

    val requestBuilder = SplitInstallRequest.newBuilder()
    NPU_DELEGATE_CANDIDATES.forEach { requestBuilder.addModule(it) }
    val request = requestBuilder.build()
    splitInstallManager.startInstall(request).addOnFailureListener { e ->
      val error = RuntimeException("NPU module installation failed to start.", e)
      Log.e(TAG, error.message, e)
      notifyNpuListeners {
        it.onFailed(
          error,
          SplitInstallSessionState.create(
            /* sessionId= */ 0,
            SplitInstallSessionStatus.FAILED,
            SplitInstallErrorCode.INTERNAL_ERROR,
            /* bytesDownloaded= */ 0,
            /* totalBytesToDownload= */ 0,
            /* moduleNames= */ request.moduleNames,
            /* languages= */ listOf(),
          ),
        )
      }
      splitInstallManager.unregisterListener(listener)
      future.complete(null)
    }
    return future
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
   * Generic helper to create a BaseOptions builder with a model buffer and an optional NPU
   * delegate.
   *
   * @param modelAiBuffer The [ByteBuffer] containing the model Ais.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @return A [BaseOptions.Builder] instance.
   */
  private fun createBaseOptions(
    modelAssetBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
  ): BaseOptions.Builder {
    val baseOptionsBuilder = BaseOptions.builder().setModelAssetBuffer(modelAssetBuffer)

    // Apply the NPU delegate if a dispatch library path is available
    if (dispatchLibraryPath != null) {
      val npuOptions =
        BaseOptions.DelegateOptions.NpuOptions.builder()
          .setDispatchLibraryDirectory(dispatchLibraryPath)
          .build()
      baseOptionsBuilder.setDelegate(Delegate.NPU).setDelegateOptions(npuOptions)
    }
    return baseOptionsBuilder
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
    creator: (Context, ByteBuffer, String?, S) -> T,
  ): Future<T> {
    val future = CompletableFuture<T>()

    npuDelegateLibraryPath.whenComplete { dispatchLibraryPath, npuException ->
      if (npuException != null) {
        future.completeExceptionally(npuException)
        return@whenComplete
      }
      val packName = "aipack_" + model.enumName

      try {
        if (aiPackManager.getPackLocation(packName) != null) {
          Log.d(TAG, "AI Pack '$packName' already installed. Creating task.")
          instantiateTaskWithModel(future, packName, model, dispatchLibraryPath, settings, creator)
        } else {
          Log.d(TAG, "AI Pack '$packName' not found. Starting download.")
          initiateAiPackDownloadForTask(
            future,
            packName,
            model,
            dispatchLibraryPath,
            settings,
            creator,
          )
        }
      } catch (e: Exception) {
        future.completeExceptionally(e)
      }
    }
    return future
  }

  /**
   * Initiates the download process for a given AI Pack and creates the task on completion.
   *
   * @param future The [CompletableFuture] to complete with the created task instance.
   * @param packName The name of the AI Pack to download.
   * @param model The [VisionModel] associated with the AI Pack.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The task-specific internal settings.
   * @param creator A lambda function that creates the task instance.
   */
  private fun <T, S> initiateAiPackDownloadForTask(
    future: CompletableFuture<T>,
    packName: String,
    model: VisionModel,
    dispatchLibraryPath: String?,
    settings: S,
    creator: (Context, ByteBuffer, String?, S) -> T,
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
            instantiateTaskWithModel(
              future,
              packName,
              model,
              dispatchLibraryPath,
              settings,
              creator,
            )
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
    dispatchLibraryPath: String?,
    settings: S,
    creator: (Context, ByteBuffer, String?, S) -> T,
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
          val task = creator(context, modelBuffer, dispatchLibraryPath, settings)
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
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [ImageSegmenter].
   * @return An [ImageSegmenter] instance.
   */
  private fun createImageSegmenterImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: ImageSegmenterSettingsInternal,
  ): ImageSegmenter {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
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
  protected fun createImageSegmenterImpl(
    model: VisionModel,
    settings: ImageSegmenterSettingsInternal,
  ): Future<ImageSegmenter> = createTask(model, settings, ::createImageSegmenterImplHelper)

  /**
   * Creates a [FaceDetector] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [FaceDetector].
   * @return A [FaceDetector] instance.
   */
  private fun createFaceDetectorImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: FaceDetectorSettingsInternal,
  ): FaceDetector {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return FaceDetector.createFromOptions(context, options)
  }

  /**
   * Creates an [FaceDetector] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [FaceDetector].
   * @return A [Future] that completes with the created [FaceDetector].
   */
  protected fun createFaceDetectorImpl(
    model: VisionModel,
    settings: FaceDetectorSettingsInternal,
  ): Future<FaceDetector> = createTask(model, settings, ::createFaceDetectorImplHelper)

  /**
   * Creates a [FaceLandmarker] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [FaceLandmarker].
   * @return A [FaceLandmarker] instance.
   */
  private fun createFaceLandmarkerImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: FaceLandmarkerSettingsInternal,
  ): FaceLandmarker {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return FaceLandmarker.createFromOptions(context, options)
  }

  /**
   * Creates an [FaceLandmarker] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [FaceLandmarker].
   * @return A [Future] that completes with the created [FaceLandmarker].
   */
  protected fun createFaceLandmarkerImpl(
    model: VisionModel,
    settings: FaceLandmarkerSettingsInternal,
  ): Future<FaceLandmarker> = createTask(model, settings, ::createFaceLandmarkerImplHelper)

  /**
   * Creates a [GestureRecognizer] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [GestureRecognizer].
   * @return A [GestureRecognizer] instance.
   */
  private fun createGestureRecognizerImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: GestureRecognizerSettingsInternal,
  ): GestureRecognizer {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return GestureRecognizer.createFromOptions(context, options)
  }

  /**
   * Creates an [GestureRecognizer] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [GestureRecognizer].
   * @return A [Future] that completes with the created [GestureRecognizer].
   */
  protected fun createGestureRecognizerImpl(
    model: VisionModel,
    settings: GestureRecognizerSettingsInternal,
  ): Future<GestureRecognizer> = createTask(model, settings, ::createGestureRecognizerImplHelper)

  /**
   * Creates an [ImageClassifier] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [ImageClassifier].
   * @return An [ImageClassifier] instance.
   */
  private fun createImageClassifierImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: ImageClassifierSettingsInternal,
  ): ImageClassifier {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return ImageClassifier.createFromOptions(context, options)
  }

  /**
   * Creates an [ImageClassifier] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [ImageClassifier].
   * @return A [Future] that completes with the created [ImageClassifier].
   */
  protected fun createImageClassifierImpl(
    model: VisionModel,
    settings: ImageClassifierSettingsInternal,
  ): Future<ImageClassifier> = createTask(model, settings, ::createImageClassifierImplHelper)

  /**
   * Creates an [ImageEmbedder] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [ImageEmbedder].
   * @return An [ImageEmbedder] instance.
   */
  private fun createImageEmbedderImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: ImageEmbedderSettingsInternal,
  ): ImageEmbedder {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return ImageEmbedder.createFromOptions(context, options)
  }

  /**
   * Creates an [ImageEmbedder] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [ImageEmbedder].
   * @return A [Future] that completes with the created [ImageEmbedder].
   */
  protected fun createImageEmbedderImpl(
    model: VisionModel,
    settings: ImageEmbedderSettingsInternal,
  ): Future<ImageEmbedder> = createTask(model, settings, ::createImageEmbedderImplHelper)

  /**
   * Creates an [InteractiveSegmenter] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [InteractiveSegmenter].
   * @return An [InteractiveSegmenter] instance.
   */
  private fun createInteractiveSegmenterImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: InteractiveSegmenterSettingsInternal,
  ): InteractiveSegmenter {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return InteractiveSegmenter.createFromOptions(context, options)
  }

  /**
   * Creates an [InteractiveSegmenter] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [InteractiveSegmenter].
   * @return A [Future] that completes with the created [InteractiveSegmenter].
   */
  protected fun createInteractiveSegmenterImpl(
    model: VisionModel,
    settings: InteractiveSegmenterSettingsInternal,
  ): Future<InteractiveSegmenter> =
    createTask(model, settings, ::createInteractiveSegmenterImplHelper)

  /**
   * Creates an [ObjectDetector] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [ObjectDetector].
   * @return An [ObjectDetector] instance.
   */
  private fun createObjectDetectorImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: ObjectDetectorSettingsInternal,
  ): ObjectDetector {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return ObjectDetector.createFromOptions(context, options)
  }

  /**
   * Creates an [ObjectDetector] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [ObjectDetector].
   * @return A [Future] that completes with the created [ObjectDetector].
   */
  protected fun createObjectDetectorImpl(
    model: VisionModel,
    settings: ObjectDetectorSettingsInternal,
  ): Future<ObjectDetector> = createTask(model, settings, ::createObjectDetectorImplHelper)

  /**
   * Creates a [PoseLandmarker] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [PoseLandmarker].
   * @return A [PoseLandmarker] instance.
   */
  private fun createPoseLandmarkerImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: PoseLandmarkerSettingsInternal,
  ): PoseLandmarker {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return PoseLandmarker.createFromOptions(context, options)
  }

  /**
   * Creates a [PoseLandmarker] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [PoseLandmarker].
   * @return A [Future] that completes with the created [PoseLandmarker].
   */
  protected fun createPoseLandmarkerImpl(
    model: VisionModel,
    settings: PoseLandmarkerSettingsInternal,
  ): Future<PoseLandmarker> = createTask(model, settings, ::createPoseLandmarkerImplHelper)

  /**
   * Creates a [HandLandmarker] instance.
   *
   * @param context The application context.
   * @param modelBuffer The [ByteBuffer] containings the model.
   * @param dispatchLibraryPath The optional path to the NPU dispatch library. If null, no NPU
   *   delegate is used.
   * @param settings The internal settings for the [HandLandmarker].
   * @return A [HandLandmarker] instance.
   */
  private fun createHandLandmarkerImplHelper(
    context: Context,
    modelBuffer: ByteBuffer,
    dispatchLibraryPath: String?,
    settings: HandLandmarkerSettingsInternal,
  ): HandLandmarker {
    val baseOptions = createBaseOptions(modelBuffer, dispatchLibraryPath).build()
    val options = settings.toOptions(baseOptions)
    return HandLandmarker.createFromOptions(context, options)
  }

  /**
   * Creates an [HandLandmarker] asynchronously.
   *
   * @param model The [VisionModel] to use.
   * @param settings The internal settings for [HandLandmarker].
   * @return A [Future] that completes with the created [HandLandmarker].
   */
  protected fun createHandLandmarkerImpl(
    model: VisionModel,
    settings: HandLandmarkerSettingsInternal,
  ): Future<HandLandmarker> = createTask(model, settings, ::createHandLandmarkerImplHelper)

  /**
   * Closes the [VisionProviderBase] and releases any resources.
   *
   * This method shuts down the internal executor.
   */
  override fun close() {
    executor.shutdown()
  }

  companion object {
    private const val TAG = "VisionProvider"
    private val NPU_DELEGATE_CANDIDATES =
      listOf("qnn_v79", "qnn_v75", "qnn_v73", "qnn_v69", "qnn_v68")
  }
}
