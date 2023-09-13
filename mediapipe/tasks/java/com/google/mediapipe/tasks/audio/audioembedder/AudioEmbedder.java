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

package com.google.mediapipe.tasks.audio.audioembedder;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.audio.audioembedder.proto.AudioEmbedderGraphOptionsProto;
import com.google.mediapipe.tasks.audio.core.BaseAudioTaskApi;
import com.google.mediapipe.tasks.audio.core.RunningMode;
import com.google.mediapipe.tasks.components.containers.AudioData;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import com.google.mediapipe.tasks.components.processors.proto.EmbedderOptionsProto;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.OutputHandler.PureResultListener;
import com.google.mediapipe.tasks.core.OutputHandler.ResultListener;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs audio embedding extraction on audio clips or audio stream.
 *
 * <p>This API expects a TFLite model with mandatory TFLite Model Metadata that contains the
 * mandatory AudioProperties of the solo input audio tensor and the optional (but recommended) label
 * items as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
 *
 * <p>Input tensor: (kTfLiteFloat32)
 *
 * <ul>
 *   <li>input audio buffer of size `[batch * samples]`.
 *   <li>batch inference is not supported (`batch` is required to be 1).
 *   <li>for multi-channel models, the channels need be interleaved.
 * </ul>
 *
 * <p>At least one output tensor with: (kTfLiteFloat32)
 *
 * <ul>
 *   <li>`N` components corresponding to the `N` dimensions of the returned feature vector for this
 *       output layer.
 *   <li>Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
 * </ul>
 */
public final class AudioEmbedder extends BaseAudioTaskApi {
  private static final String TAG = AudioEmbedder.class.getSimpleName();
  private static final String AUDIO_IN_STREAM_NAME = "audio_in";
  private static final String SAMPLE_RATE_IN_STREAM_NAME = "sample_rate_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "AUDIO:" + AUDIO_IN_STREAM_NAME, "SAMPLE_RATE:" + SAMPLE_RATE_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "EMBEDDINGS:embeddings_out", "TIMESTAMPED_EMBEDDINGS:timestamped_embeddings_out"));
  private static final int EMBEDDINGS_OUT_STREAM_INDEX = 0;
  private static final int TIMESTAMPED_EMBEDDINGS_OUT_STREAM_INDEX = 1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.audio.audio_embedder.AudioEmbedderGraph";
  private static final long MICROSECONDS_PER_MILLISECOND = 1000;

  static {
    ProtoUtil.registerTypeName(
        EmbeddingsProto.EmbeddingResult.class,
        "mediapipe.tasks.components.containers.proto.EmbeddingResult");
  }

  /**
   * Creates an {@link AudioEmbedder} instance from a model file and default {@link
   * AudioEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the embedding model in the assets.
   * @throws MediaPipeException if there is an error during {@link AudioEmbedder} creation.
   */
  public static AudioEmbedder createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, AudioEmbedderOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link AudioEmbedder} instance from a model file and default {@link
   * AudioEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the embedding model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link AudioEmbedder} creation.
   */
  public static AudioEmbedder createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, AudioEmbedderOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates an {@link AudioEmbedder} instance from a model buffer and default {@link
   * AudioEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the embedding
   *     model.
   * @throws MediaPipeException if there is an error during {@link AudioEmbedder} creation.
   */
  public static AudioEmbedder createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, AudioEmbedderOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link AudioEmbedder} instance from an {@link AudioEmbedderOptions} instance.
   *
   * @param context an Android {@link Context}.
   * @param options an {@link AudioEmbedderOptions} instance.
   * @throws MediaPipeException if there is an error during {@link AudioEmbedder} creation.
   */
  public static AudioEmbedder createFromOptions(Context context, AudioEmbedderOptions options) {
    OutputHandler<AudioEmbedderResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<AudioEmbedderResult, Void>() {
          @Override
          public AudioEmbedderResult convertToTaskResult(List<Packet> packets) {
            try {
              if (!packets.get(EMBEDDINGS_OUT_STREAM_INDEX).isEmpty()) {
                // For audio stream mode.
                return AudioEmbedderResult.createFromProto(
                    PacketGetter.getProto(
                        packets.get(EMBEDDINGS_OUT_STREAM_INDEX),
                        EmbeddingsProto.EmbeddingResult.getDefaultInstance()),
                    packets.get(EMBEDDINGS_OUT_STREAM_INDEX).getTimestamp()
                        / MICROSECONDS_PER_MILLISECOND);
              } else {
                // For audio clips mode.
                return AudioEmbedderResult.createFromProtoList(
                    PacketGetter.getProtoVector(
                        packets.get(TIMESTAMPED_EMBEDDINGS_OUT_STREAM_INDEX),
                        EmbeddingsProto.EmbeddingResult.parser()),
                    -1);
              }
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
    if (options.resultListener().isPresent()) {
      ResultListener<AudioEmbedderResult, Void> resultListener =
          new ResultListener<AudioEmbedderResult, Void>() {
            @Override
            public void run(AudioEmbedderResult audioEmbedderResult, Void input) {
              options.resultListener().get().run(audioEmbedderResult);
            }
          };
      handler.setResultListener(resultListener);
    }
    options.errorListener().ifPresent(handler::setErrorListener);
    // Audio tasks should not drop input audio due to flow limiting, which may cause data
    // inconsistency.
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<AudioEmbedderOptions>builder()
                .setTaskName(AudioEmbedder.class.getSimpleName())
                .setTaskRunningModeName(options.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new AudioEmbedder(runner, options.runningMode());
  }

  /**
   * Constructor to initialize an {@link AudioEmbedder} from a {@link TaskRunner} and {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe audio task {@link RunningMode}.
   */
  private AudioEmbedder(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, AUDIO_IN_STREAM_NAME, SAMPLE_RATE_IN_STREAM_NAME);
  }

  /*
   * Performs embedding extraction on the provided audio clips. Only use this method when the
   * AudioEmbedder is created with the audio clips mode.
   *
   * <p>The audio clip is represented as a MediaPipe {@link AudioData} object  The method accepts
   * audio clips with various length and audio sample rate.  It's required to provide the
   * corresponding audio sample rate within the {@link AudioData} object.
   *
   * <p>The input audio clip may be longer than what the model is able to process in a single
   * inference. When this occurs, the input audio clip is split into multiple chunks starting at
   * different timestamps. For this reason, this function returns a vector of EmbeddingResult
   * objects, each associated with a timestamp corresponding to the start (in milliseconds) of the
   * chunk data that was extracted.
   *
   * @param audioClip a MediaPipe {@link AudioData} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public AudioEmbedderResult embed(AudioData audioClip) {
    return (AudioEmbedderResult) processAudioClip(audioClip);
  }

  /*
   * Sends audio data (a block in a continuous audio stream) to perform audio embedding, and
   * the results will be available via the {@link ResultListener} provided in the
   * {@link AudioClassifierOptions}. Only use this method when the AudioEmbedder is created with
   * the audio stream mode.
   *
   * <p>The audio block is represented as a MediaPipe {@link AudioData} object. The audio data will
   * be resampled, accumulated, and framed to the proper size for the underlying model to consume.
   * It's required to provide the corresponding audio sample rate within {@link AudioData} object as
   * well as a timestamp (in milliseconds) to indicate the start time of the input audio block. The
   * timestamps must be monotonically increasing. This method will return immediately after
   * the input audio data is accepted. The results will be available in the `resultListener`
   * provided in the `AudioEmbedderOptions`. The `embedAsync` method is designed to process
   * auido stream data such as microphone input.
   *
   * <p>The input audio block may be longer than what the model is able to process in a single
   * inference. When this occurs, the input audio block is split into multiple chunks. For this
   * reason, the callback may be called multiple times (once per chunk) for each call to this
   * function.
   *
   * @param audioBlock a MediaPipe {@link AudioData} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void embedAsync(AudioData audioBlock, long timestampMs) {
    checkOrSetSampleRate(audioBlock.getFormat().getSampleRate());
    sendAudioStreamData(audioBlock, timestampMs);
  }

  /** Options for setting up and {@link AudioEmbedder}. */
  @AutoValue
  public abstract static class AudioEmbedderOptions extends TaskOptions {

    /** Builder for {@link AudioEmbedderOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the audio embedder task. */
      public abstract Builder setBaseOptions(BaseOptions baseOptions);

      /**
       * Sets the {@link RunningMode} for the audio embedder task. Default to the audio clips mode.
       * Image embedder has two modes:
       *
       * <ul>
       *   <li>AUDIO_CLIPS: The mode for running audio embedding on audio clips. Users feed audio
       *       clips to the `embed` method, and will receive the embedding results as the return
       *       value.
       *   <li>AUDIO_STREAM: The mode for running audio embedding on the audio stream, such as from
       *       microphone. Users call `embedAsync` to push the audio data into the AudioEmbedder,
       *       the embedding results will be available in the result callback when the audio
       *       embedder finishes the work.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode runningMode);

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

      /**
       * Sets the {@link ResultListener} to receive the embedding results asynchronously when the
       * audio embedder is in the audio stream mode.
       */
      public abstract Builder setResultListener(
          PureResultListener<AudioEmbedderResult> resultListener);

      /** Sets an optional {@link ErrorListener}. */
      public abstract Builder setErrorListener(ErrorListener errorListener);

      abstract AudioEmbedderOptions autoBuild();

      /**
       * Validates and builds the {@link AudioEmbedderOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the audio embedder is
       *     in the audio stream mode.
       */
      public final AudioEmbedderOptions build() {
        AudioEmbedderOptions options = autoBuild();
        if (options.runningMode() == RunningMode.AUDIO_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The audio embedder is in the audio stream mode, a user-defined result listener"
                    + " must be provided in the AudioEmbedderOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The audio embedder is in the audio clips mode, a user-defined result listener"
                  + " shouldn't be provided in AudioEmbedderOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract boolean l2Normalize();

    abstract boolean quantize();

    abstract Optional<PureResultListener<AudioEmbedderResult>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_AudioEmbedder_AudioEmbedderOptions.Builder()
          .setRunningMode(RunningMode.AUDIO_CLIPS)
          .setL2Normalize(false)
          .setQuantize(false);
    }

    /** Converts a {@link AudioEmbedderOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() == RunningMode.AUDIO_STREAM);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      EmbedderOptionsProto.EmbedderOptions.Builder embedderOptionsBuilder =
          EmbedderOptionsProto.EmbedderOptions.newBuilder();
      embedderOptionsBuilder.setL2Normalize(l2Normalize());
      embedderOptionsBuilder.setQuantize(quantize());
      AudioEmbedderGraphOptionsProto.AudioEmbedderGraphOptions.Builder taskOptionsBuilder =
          AudioEmbedderGraphOptionsProto.AudioEmbedderGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setEmbedderOptions(embedderOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              AudioEmbedderGraphOptionsProto.AudioEmbedderGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
