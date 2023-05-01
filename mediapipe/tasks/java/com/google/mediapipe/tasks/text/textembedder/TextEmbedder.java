// Copyright 2022 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.text.textembedder;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.components.containers.Embedding;
import com.google.mediapipe.tasks.components.containers.EmbeddingResult;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import com.google.mediapipe.tasks.components.processors.proto.EmbedderOptionsProto;
import com.google.mediapipe.tasks.components.utils.CosineSimilarity;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.text.textembedder.proto.TextEmbedderGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Performs embedding extraction on text.
 *
 * <p>This API expects a TFLite model with (optional) <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata</a>.
 *
 * <p>Metadata is required for models with int32 input tensors because it contains the input process
 * unit for the model's Tokenizer. No metadata is required for models with string input tensors.
 *
 * <ul>
 *   <li>Input tensors
 *       <ul>
 *         <li>Three input tensors ({@code kTfLiteInt32}) of shape {@code [batch_size x
 *             bert_max_seq_len]} representing the input ids, mask ids, and segment ids. This input
 *             signature requires a Bert Tokenizer process unit in the model metadata.
 *         <li>Or one input tensor ({@code kTfLiteInt32}) of shape {@code [batch_size x
 *             max_seq_len]} representing the input ids. This input signature requires a Regex
 *             Tokenizer process unit in the model metadata.
 *         <li>Or one input tensor ({@code kTfLiteString}) that is shapeless or has shape {@code
 *             [1]} containing the input string.
 *       </ul>
 *   <li>At least one output tensor ({@code kTfLiteFloat32}/{@code kTfLiteUint8}) with shape {@code
 *       [1 x N]} where N is the number of dimensions in the produced embeddings.
 * </ul>
 */
public final class TextEmbedder implements AutoCloseable {
  private static final String TAG = TextEmbedder.class.getSimpleName();
  private static final String TEXT_IN_STREAM_NAME = "text_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("TEXT:" + TEXT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("EMBEDDINGS:embeddings_out"));

  private static final int EMBEDDINGS_OUT_STREAM_INDEX = 0;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.text.text_embedder.TextEmbedderGraph";
  private final TaskRunner runner;

  static {
    System.loadLibrary("mediapipe_tasks_text_jni");
    ProtoUtil.registerTypeName(
        EmbeddingsProto.EmbeddingResult.class,
        "mediapipe.tasks.components.containers.proto.EmbeddingResult");
  }

  /**
   * Creates a {@link TextEmbedder} instance from a model file and the default {@link
   * TextEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the text model with metadata in the assets.
   * @throws MediaPipeException if there is is an error during {@link TextEmbedder} creation.
   */
  public static TextEmbedder createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, TextEmbedderOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link TextEmbedder} instance from a model file and the default {@link
   * TextEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the text model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link TextEmbedder} creation.
   */
  public static TextEmbedder createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, TextEmbedderOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link TextEmbedder} instance from {@link TextEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param options a {@link TextEmbedderOptions} instance.
   * @throws MediaPipeException if there is an error during {@link TextEmbedder} creation.
   */
  public static TextEmbedder createFromOptions(Context context, TextEmbedderOptions options) {
    OutputHandler<TextEmbedderResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<TextEmbedderResult, Void>() {
          @Override
          public TextEmbedderResult convertToTaskResult(List<Packet> packets) {
            try {
              return TextEmbedderResult.create(
                  EmbeddingResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(EMBEDDINGS_OUT_STREAM_INDEX),
                          EmbeddingsProto.EmbeddingResult.getDefaultInstance())),
                  packets.get(EMBEDDINGS_OUT_STREAM_INDEX).getTimestamp());
            } catch (IOException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(), e.getMessage());
            }
          }

          @Override
          public Void convertToTaskInput(List<Packet> packets) {
            return null;
          }
        });
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<TextEmbedderOptions>builder()
                .setTaskName(TextEmbedder.class.getSimpleName())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new TextEmbedder(runner);
  }

  /**
   * Constructor to initialize a {@link TextEmbedder} from a {@link TaskRunner}.
   *
   * @param runner a {@link TaskRunner}.
   */
  private TextEmbedder(TaskRunner runner) {
    this.runner = runner;
  }

  /**
   * Performs embedding extraction on the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public TextEmbedderResult embed(String inputText) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(TEXT_IN_STREAM_NAME, runner.getPacketCreator().createString(inputText));
    return (TextEmbedderResult) runner.process(inputPackets);
  }

  /** Closes and cleans up the {@link TextEmbedder}. */
  @Override
  public void close() {
    runner.close();
  }

  /**
   * Utility function to compute <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine
   * similarity</a> between two {@link Embedding} objects.
   *
   * @throws IllegalArgumentException if the embeddings are of different types (float vs.
   *     quantized), have different sizes, or have an L2-norm of 0.
   */
  public static double cosineSimilarity(Embedding u, Embedding v) {
    return CosineSimilarity.compute(u, v);
  }

  /** Options for setting up a {@link TextEmbedder}. */
  @AutoValue
  public abstract static class TextEmbedderOptions extends TaskOptions {

    /** Builder for {@link TextEmbedderOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the text embedder task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets whether L2 normalization should be performed on the returned embeddings. Use this
       * option only if the model does not already contain a native <code>L2_NORMALIZATION</code> TF
       * Lite Op. In most cases, this is already the case and L2 norm is thus achieved through TF
       * Lite inference.
       *
       * <p>False by default.
       */
      public abstract Builder setL2Normalize(boolean l2Normalize);

      /**
       * Sets whether the returned embedding should be quantized to bytes via scalar quantization.
       * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is
       * guaranteed to have value in <code>[-1.0, 1.0]</code>. Use {@link #setL2Normalize(boolean)}
       * if this is not the case.
       *
       * <p>False by default.
       */
      public abstract Builder setQuantize(boolean quantize);

      public abstract TextEmbedderOptions build();
    }

    abstract BaseOptions baseOptions();

    abstract boolean l2Normalize();

    abstract boolean quantize();

    public static Builder builder() {
      return new AutoValue_TextEmbedder_TextEmbedderOptions.Builder()
          .setL2Normalize(false)
          .setQuantize(false);
    }

    /** Converts a {@link TextEmbedderOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      EmbedderOptionsProto.EmbedderOptions.Builder embedderOptionsBuilder =
          EmbedderOptionsProto.EmbedderOptions.newBuilder();
      embedderOptionsBuilder.setL2Normalize(l2Normalize());
      embedderOptionsBuilder.setQuantize(quantize());
      TextEmbedderGraphOptionsProto.TextEmbedderGraphOptions.Builder taskOptionsBuilder =
          TextEmbedderGraphOptionsProto.TextEmbedderGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setEmbedderOptions(embedderOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              TextEmbedderGraphOptionsProto.TextEmbedderGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
