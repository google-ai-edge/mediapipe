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

package com.google.mediapipe.tasks.vision.imageembedder;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.EmbeddingEngine;
import com.google.ai.edge.litertlm.EmbeddingEngineConfig;
import com.google.ai.edge.litertlm.EmbeddingOptions;
import com.google.ai.edge.litertlm.EmbeddingResponse;
import com.google.ai.edge.litertlm.InputData;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.mediapipe.calculator.proto.InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Embedding;
import com.google.mediapipe.tasks.components.containers.EmbeddingResult;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLoggerFactory;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder.ImageEmbedderOptions;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Collections;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Helper implementation class for LiteRT-LM Image Embedder integration. */
public final class ImageEmbedderLiteRtLmExecutorImpl implements ImageEmbedderExecutor {
  private static final String TAG = "ImageEmbedderLiteRtLm";
  private final EmbeddingEngine engine;
  private final TasksStatsLogger statsLogger;
  private final ExecutorService executorService;
  private final ImageEmbedderOptions options;
  private long lastSeenTimestamp = Long.MIN_VALUE;

  public ImageEmbedderLiteRtLmExecutorImpl(Context context, ImageEmbedderOptions options) {
    this.options = options;
    this.engine = createLiteRtLmEngine(context, options);
    this.statsLogger =
        TasksStatsLoggerFactory.create(
            context, ImageEmbedder.class.getSimpleName(), options.runningMode().name());
    this.statsLogger.logSessionStart();
    if (options.runningMode() == RunningMode.LIVE_STREAM) {
      this.executorService =
          Executors.newSingleThreadExecutor(
              new ThreadFactoryBuilder().setNameFormat("image-embedder-litertlm-%d").build());
    } else {
      this.executorService = null;
    }
  }

  private static EmbeddingEngine createLiteRtLmEngine(
      Context context, ImageEmbedderOptions options) {
    BaseOptionsProto.BaseOptions proto =
        BaseOptionsUtils.convertBaseOptionsToProto(options.baseOptions());

    if (proto.getModelAsset().hasFilePointerMeta() || proto.getModelAsset().hasFileContent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT,
          "Loading model from memory buffer is not supported by LiteRT-LM.",
          null);
    }

    if (proto.getModelAsset().hasFileDescriptorMeta()
        && proto.getModelAsset().getFileDescriptorMeta().getFd() > 0) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT,
          "Loading model from file descriptor is not supported by LiteRT-LM.",
          null);
    }

    String finalPath = null;
    Integer fd = null;
    if (proto.getModelAsset().hasFileDescriptorMeta()
        && proto.getModelAsset().getFileDescriptorMeta().getFd() > 0) {
      fd = proto.getModelAsset().getFileDescriptorMeta().getFd();
    } else if (!proto.getModelAsset().getFileName().isEmpty()) {
      finalPath = proto.getModelAsset().getFileName();
    } else {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(),
          "Model path or file descriptor must be specified in BaseOptions.");
    }

    Backend backend = getBackend(context, proto);

    EmbeddingEngineConfig config =
        new EmbeddingEngineConfig(
            finalPath, backend, backend, backend, context.getCacheDir().getAbsolutePath(), fd);
    EmbeddingEngine engine = new EmbeddingEngine(config);
    engine.initialize();
    return engine;
  }

  @Override
  public ImageEmbedderResult embed(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    if (options.runningMode() != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION,
          "Task is not initialized with the image mode. Current running mode: "
              + options.runningMode().name(),
          null);
    }
    long timestamp = generateSyntheticTimestamp();
    statsLogger.recordCpuInputArrival(timestamp);
    EmbeddingResult embeddingResult = computeEmbedding(image);
    statsLogger.recordInvocationEnd(timestamp);
    return ImageEmbedderResult.create(embeddingResult, timestamp / 1000);
  }

  @Override
  public ImageEmbedderResult embedForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    if (options.runningMode() != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION,
          "Task is not initialized with the video mode. Current running mode: "
              + options.runningMode().name(),
          null);
    }
    long timestampUs = timestampMs * 1000;
    statsLogger.recordCpuInputArrival(timestampUs);
    EmbeddingResult embeddingResult = computeEmbedding(image);
    statsLogger.recordInvocationEnd(timestampUs);
    return ImageEmbedderResult.create(embeddingResult, timestampMs);
  }

  @Override
  public void embedAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    if (options.runningMode() != RunningMode.LIVE_STREAM) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION,
          "Task is not initialized with the live stream mode. Current running mode: "
              + options.runningMode().name(),
          null);
    }
    long timestampUs = timestampMs * 1000;
    statsLogger.recordCpuInputArrival(timestampUs);

    if (executorService != null) {
      executorService.execute(
          () -> {
            try {
              EmbeddingResult embeddingResult = computeEmbedding(image);
              statsLogger.recordInvocationEnd(timestampUs);
              ImageEmbedderResult result = ImageEmbedderResult.create(embeddingResult, timestampMs);
              if (options.resultListener().isPresent()) {
                options.resultListener().get().run(result, image);
              }
            } catch (RuntimeException e) {
              if (options.errorListener().isPresent()) {
                MediaPipeException exception =
                    new MediaPipeException(
                        MediaPipeException.StatusCode.INTERNAL, e.getMessage(), e);
                options.errorListener().get().onError(exception);
              } else {
                Log.e(TAG, "Error in embedAsync: ", e);
              }
            }
          });
    }
  }

  private EmbeddingResult computeEmbedding(MPImage image) {
    Bitmap bitmap = BitmapExtractor.extract(image);
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    bitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream);
    ImmutableList<InputData> contents = ImmutableList.of(new InputData.Image(stream.toByteArray()));

    EmbeddingOptions embeddingOptions = new EmbeddingOptions(options.l2Normalize(), null);
    EmbeddingResponse response = engine.computeEmbedding(contents, embeddingOptions);

    Embedding embedding;
    if (options.quantize()) {
      float[] floatEmbedding = response.getEmbedding();
      byte[] quantizedEmbedding = new byte[floatEmbedding.length];
      for (int i = 0; i < floatEmbedding.length; i++) {
        int unclampedValue = Math.round(floatEmbedding[i] * 128.0f);
        quantizedEmbedding[i] = (byte) Math.max(-128, Math.min(unclampedValue, 127));
      }
      embedding = Embedding.create(new float[0], quantizedEmbedding, 0, Optional.empty());
    } else {
      embedding = Embedding.create(response.getEmbedding(), new byte[0], 0, Optional.empty());
    }
    return EmbeddingResult.create(Collections.singletonList(embedding), Optional.empty());
  }

  @Override
  public void close() {
    engine.close();
    if (executorService != null) {
      executorService.shutdown();
    }
    statsLogger.logSessionEnd();
  }

  private synchronized long generateSyntheticTimestamp() {
    long timestamp = lastSeenTimestamp == Long.MIN_VALUE ? 0 : lastSeenTimestamp + 1000000;
    lastSeenTimestamp = timestamp;
    return timestamp;
  }

  private static Backend getBackend(Context context, BaseOptionsProto.BaseOptions proto) {
    if (proto == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT,
          "BaseOptions proto cannot be null.",
          null);
    }

    try {
      if (proto.getAcceleration().hasGpu()) {
        return new Backend.GPU();
      }

      if (proto.getAcceleration().hasLitert()) {
        LiteRt litert = proto.getAcceleration().getLitert();

        if (litert.hasNpu()) {
          String dispatchDir =
              context.getApplicationInfo() != null
                      && context.getApplicationInfo().nativeLibraryDir != null
                  ? context.getApplicationInfo().nativeLibraryDir
                  : "";
          String customPath = litert.getNpu().getDispatchLibraryPath();
          if (!customPath.isEmpty()) {
            File customDir = new File(customPath);
            if (!customDir.exists()) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INVALID_ARGUMENT,
                  "NPU dispatch library directory does not exist: " + customPath,
                  null);
            }
            dispatchDir = customPath;
          }
          return new Backend.NPU(dispatchDir);
        }

        if (litert.hasGpu()) {
          return new Backend.GPU();
        }

        if (litert.hasCpu()) {
          return new Backend.CPU();
        }

        return new Backend.CPU();
      }

      return new Backend.CPU();
    } catch (MediaPipeException e) {
      throw e;
    } catch (RuntimeException e) {
      MediaPipeException exception =
          new MediaPipeException(
              MediaPipeException.StatusCode.INTERNAL,
              "Failed to configure backend execution engine: " + e.getMessage(),
              e);
      throw exception;
    }
  }
}
