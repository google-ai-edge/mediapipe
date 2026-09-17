/* Copyright 2026 The MediaPipe Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.tasks.retrieval.universalembedder;

import static java.nio.charset.StandardCharsets.UTF_8;

import android.content.Context;
import android.graphics.Bitmap;
import com.google.ai.edge.litertlm.Backend;
import com.google.ai.edge.litertlm.EmbeddingEngine;
import com.google.ai.edge.litertlm.EmbeddingEngineConfig;
import com.google.ai.edge.litertlm.EmbeddingOptions;
import com.google.ai.edge.litertlm.EmbeddingResponse;
import com.google.ai.edge.litertlm.InputData;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.calculator.proto.InferenceCalculatorProto.InferenceCalculatorOptions.Delegate.LiteRt;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.AudioData;
import com.google.mediapipe.tasks.components.containers.Embedding;
import com.google.mediapipe.tasks.components.containers.EmbeddingResult;
import com.google.mediapipe.tasks.components.utils.CosineSimilarity;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.core.EmbeddingProvider;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLoggerFactory;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * UniversalEmbedder combines the features of the existing Audio, Text, and Image embedders, while
 * adding support for video embeddings. It wraps the native LiteRT-LM EmbeddingEngine directly,
 * providing real on-device multimodal embeddings.
 */
@SuppressWarnings("EnumOrdinal")
public final class UniversalEmbedder implements AutoCloseable {

  private final UniversalEmbedderOptions options;
  private final EmbeddingEngine engine;
  private final TasksStatsLogger statsLogger;
  private final AtomicLong syntheticTimestamp = new AtomicLong(0);

  private UniversalEmbedder(Context context, UniversalEmbedderOptions options) {
    this.options = options;
    this.statsLogger =
        TasksStatsLoggerFactory.create(context, "UniversalEmbedder", /* taskRunningModeStr= */ "");
    this.statsLogger.logSessionStart();

    BaseOptionsProto.BaseOptions proto =
        BaseOptionsUtils.convertBaseOptionsToProto(options.baseOptions());

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

    Backend defaultBackend = getBackend(context, proto);
    Backend backend =
        options.textDelegate().isPresent()
            ? getBackendForDelegate(context, options.textDelegate().get(), defaultBackend)
            : defaultBackend;
    Backend visionBackend =
        options.visionDelegate().isPresent()
            ? getBackendForDelegate(context, options.visionDelegate().get(), defaultBackend)
            : defaultBackend;
    Backend audioBackend =
        options.audioDelegate().isPresent()
            ? getBackendForDelegate(context, options.audioDelegate().get(), defaultBackend)
            : defaultBackend;

    Integer maxInputLength = options.maxInputLength().orElse(null);
    Integer visionTokensPerImage = options.visionTokensPerImage().orElse(null);

    String cacheDir = options.cacheDir().orElseGet(() -> context.getCacheDir().getAbsolutePath());

    EmbeddingEngineConfig config =
        new EmbeddingEngineConfig(
            finalPath,
            backend,
            visionBackend,
            audioBackend,
            cacheDir,
            fd,
            maxInputLength,
            visionTokensPerImage,
            options
                .activationDataType()
                .map(UniversalEmbedderOptions.ActivationDataType::toLiteRtLmType)
                .orElse(null));
    this.engine = new EmbeddingEngine(config);
    this.engine.initialize();
  }

  private static Backend getBackendForDelegate(
      Context context, Delegate delegate, Backend mainBackend) {
    switch (delegate) {
      case GPU:
        return new Backend.GPU();
      case NPU:
        String dispatchDir =
            mainBackend instanceof Backend.NPU
                ? ((Backend.NPU) mainBackend).getNativeLibraryDir()
                : (context.getApplicationInfo() != null
                        && context.getApplicationInfo().nativeLibraryDir != null
                    ? context.getApplicationInfo().nativeLibraryDir
                    : "");
        return new Backend.NPU(dispatchDir);
      case CPU:
      default:
        return new Backend.CPU();
    }
  }

  /**
   * Evaluates the selected acceleration configurations inside {@link BaseOptions} and maps them to
   * a native LiteRT-LM {@link Backend} with robust validation.
   */
  private static Backend getBackend(Context context, BaseOptionsProto.BaseOptions proto) {
    if (proto == null) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(),
          "BaseOptions proto cannot be null.");
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
                  MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(),
                  "NPU dispatch library directory does not exist: " + customPath);
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
              MediaPipeException.StatusCode.INTERNAL.ordinal(),
              "Failed to configure backend execution engine: " + e.getMessage());
      exception.initCause(e);
      throw exception;
    }
  }

  public static UniversalEmbedder createFromOptions(
      Context context, UniversalEmbedderOptions options) {
    return new UniversalEmbedder(context, options);
  }

  public EmbeddingResult embedText(String text) {
    ImmutableList<InputData> contents = ImmutableList.of(new InputData.Text(text));
    return executeInference(contents);
  }

  public EmbeddingResult embedImage(MPImage image) {
    Bitmap bitmap = BitmapExtractor.extract(image);
    ImmutableList<InputData> contents = ImmutableList.of(new InputData.Image(encodeAsTga(bitmap)));
    return executeInference(contents);
  }

  public EmbeddingResult embedAudio(AudioData audioData) {
    float[] buffer = audioData.getBuffer();
    AudioData.AudioDataFormat format = audioData.getFormat();
    byte[] wavBytes = encodeAsWav(buffer, (int) format.getSampleRate(), format.getNumOfChannels());
    ImmutableList<InputData> contents = ImmutableList.of(new InputData.Audio(wavBytes));
    return executeInference(contents);
  }

  /**
   * Performs embedding extraction on multi-modal parts of a content block.
   *
   * @param content the list of multi-modal content to embed.
   * @return a {@link EmbeddingResult} object that contains the embeddings.
   */
  public EmbeddingResult embedContent(List<Object> content) {
    return executeInference(convertToInputData(content));
  }

  public EmbeddingProvider getProvider() {
    return new EmbeddingProvider() {
      @Override
      public float[] embedContent(List<Object> content) {
        EmbeddingResult result = UniversalEmbedder.this.embedContent(content);
        if (result.embeddings().isEmpty()) {
          return new float[0];
        }
        return result.embeddings().get(0).floatEmbedding();
      }
    };
  }

  private List<InputData> convertToInputData(List<Object> content) {
    List<InputData> contents = new ArrayList<>();
    for (Object obj : content) {
      if (obj instanceof String) {
        contents.add(new InputData.Text((String) obj));
      } else if (obj instanceof byte[]) {
        contents.add(new InputData.Image((byte[]) obj));
      } else if (obj instanceof MPImage) {
        Bitmap bitmap = BitmapExtractor.extract((MPImage) obj);
        contents.add(new InputData.Image(encodeAsTga(bitmap)));
      } else if (obj instanceof AudioData) {
        AudioData audioData = (AudioData) obj;
        float[] buffer = audioData.getBuffer();
        AudioData.AudioDataFormat format = audioData.getFormat();
        byte[] wavBytes =
            encodeAsWav(buffer, (int) format.getSampleRate(), format.getNumOfChannels());
        contents.add(new InputData.Audio(wavBytes));
      } else {
        contents.add(new InputData.Text(String.valueOf(obj)));
      }
    }
    return contents;
  }

  private EmbeddingResult executeInference(List<InputData> contents) {
    long timestamp = syntheticTimestamp.getAndIncrement();
    statsLogger.recordCpuInputArrival(timestamp);
    EmbeddingOptions embeddingOptions = new EmbeddingOptions(options.l2Normalize());
    EmbeddingResponse response = engine.computeEmbedding(contents, embeddingOptions);
    Embedding embedding =
        Embedding.create(response.getEmbedding(), new byte[0], 0, Optional.empty());
    statsLogger.recordInvocationEnd(timestamp);
    return EmbeddingResult.create(Collections.singletonList(embedding), Optional.empty());
  }

  /** Utility function to compute cosine similarity between two {@link Embedding} objects. */
  public static double cosineSimilarity(Embedding u, Embedding v) {
    return CosineSimilarity.compute(u, v);
  }

  /** Utility function to compute cosine similarity between two float arrays. */
  public static double cosineSimilarity(float[] vectorA, float[] vectorB) {
    Embedding embeddingA = Embedding.create(vectorA, new byte[0], 0, Optional.empty());
    Embedding embeddingB = Embedding.create(vectorB, new byte[0], 0, Optional.empty());
    return CosineSimilarity.compute(embeddingA, embeddingB);
  }

  private static byte[] encodeAsWav(float[] pcmData, int sampleRate, int channels) {
    int dataSize = pcmData.length * 2;
    ByteBuffer buffer = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN);

    // 1. RIFF Header
    buffer.put("RIFF".getBytes());
    buffer.putInt(dataSize + 36);
    buffer.put("WAVEfmt ".getBytes(UTF_8));

    // 2. Format Chunk
    buffer.putInt(16); // Subchunk1Size (16 for PCM)
    buffer.putShort((short) 1); // AudioFormat (1 for PCM)
    buffer.putShort((short) channels); // NumChannels
    buffer.putInt(sampleRate); // SampleRate
    buffer.putInt(sampleRate * channels * 2); // ByteRate
    buffer.putShort((short) (channels * 2)); // BlockAlign
    buffer.putShort((short) 16); // BitsPerSample

    // 3. Data Chunk
    buffer.put("data".getBytes());
    buffer.putInt(dataSize);

    // 4. Convert and write PCM Audio Data
    for (float sample : pcmData) {
      short val = (short) Math.max(-32768, Math.min(32767, Math.round(sample * 32767.0f)));
      buffer.putShort(val);
    }

    return buffer.array();
  }

  private static byte[] encodeAsTga(Bitmap bitmap) {
    int width = bitmap.getWidth();
    int height = bitmap.getHeight();
    int[] pixels = new int[width * height];
    bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

    byte[] tgaBytes = new byte[20 + pixels.length * 4];
    ByteBuffer buffer = ByteBuffer.wrap(tgaBytes).order(ByteOrder.LITTLE_ENDIAN);

    // TGA Header (18 bytes) + 2 dummy bytes for alignment
    tgaBytes[0] = 2; // ID Length = 2 bytes (acts as padding)
    tgaBytes[2] = 2; // Uncompressed True-color
    tgaBytes[12] = (byte) (width & 0xFF);
    tgaBytes[13] = (byte) ((width >> 8) & 0xFF);
    tgaBytes[14] = (byte) (height & 0xFF);
    tgaBytes[15] = (byte) ((height >> 8) & 0xFF);
    tgaBytes[16] = 32; // 32 bits per pixel
    tgaBytes[17] = 0x28; // Top-down, 8 alpha bits

    // Clear bytes 18, 19 (Dummy ID bytes for alignment)
    tgaBytes[18] = 0;
    tgaBytes[19] = 0;

    // Fast Write Pixels
    buffer.position(20);
    buffer.asIntBuffer().put(pixels);

    return tgaBytes;
  }

  @Override
  public void close() {
    statsLogger.logSessionEnd();
    engine.close();
  }
}
