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

package com.google.mediapipe.tasks.audio.audioclassifier;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.audio.audioclassifier.proto.AudioClassifierGraphOptionsProto;
import com.google.mediapipe.tasks.audio.core.BaseAudioTaskApi;
import com.google.mediapipe.tasks.audio.core.RunningMode;
import com.google.mediapipe.tasks.components.containers.AudioData;
import com.google.mediapipe.tasks.components.containers.proto.ClassificationsProto;
import com.google.mediapipe.tasks.components.processors.proto.ClassifierOptionsProto;
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
 * Performs audio classification on audio clips or audio stream.
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
 *   <li>`[1 x N]` array with `N` represents the number of categories.
 *   <li>optional (but recommended) label items as AssociatedFiles with type TENSOR_AXIS_LABELS,
 *       containing one label per line. The first such AssociatedFile (if any) is used to fill the
 *       `category_name` field of the results. The `display_name` field is filled from the
 *       AssociatedFile (if any) whose locale matches the `display_names_locale` field of the
 *       `AudioClassifierOptions` used at creation time ("en" by default, i.e. English). If none of
 *       these are available, only the `index` field of the results will be filled.
 * </ul>
 */
public final class AudioClassifier extends BaseAudioTaskApi {
  private static final String TAG = AudioClassifier.class.getSimpleName();
  private static final String AUDIO_IN_STREAM_NAME = "audio_in";
  private static final String SAMPLE_RATE_IN_STREAM_NAME = "sample_rate_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "AUDIO:" + AUDIO_IN_STREAM_NAME, "SAMPLE_RATE:" + SAMPLE_RATE_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "CLASSIFICATIONS:classifications_out",
              "TIMESTAMPED_CLASSIFICATIONS:timestamped_classifications_out"));
  private static final int CLASSIFICATIONS_OUT_STREAM_INDEX = 0;
  private static final int TIMESTAMPED_CLASSIFICATIONS_OUT_STREAM_INDEX = 1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.audio.audio_classifier.AudioClassifierGraph";
  private static final long MICROSECONDS_PER_MILLISECOND = 1000;

  static {
    ProtoUtil.registerTypeName(
        ClassificationsProto.ClassificationResult.class,
        "mediapipe.tasks.components.containers.proto.ClassificationResult");
  }

  /**
   * Creates an {@link AudioClassifier} instance from a model file and default {@link
   * AudioClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the classification model in the assets.
   * @throws MediaPipeException if there is an error during {@link AudioClassifier} creation.
   */
  public static AudioClassifier createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, AudioClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link AudioClassifier} instance from a model file and default {@link
   * AudioClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the classification model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link AudioClassifier} creation.
   */
  public static AudioClassifier createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, AudioClassifierOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates an {@link AudioClassifier} instance from a model buffer and default {@link
   * AudioClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the
   *     classification model.
   * @throws MediaPipeException if there is an error during {@link AudioClassifier} creation.
   */
  public static AudioClassifier createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, AudioClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link AudioClassifier} instance from an {@link AudioClassifierOptions} instance.
   *
   * @param context an Android {@link Context}.
   * @param options an {@link AudioClassifierOptions} instance.
   * @throws MediaPipeException if there is an error during {@link AudioClassifier} creation.
   */
  public static AudioClassifier createFromOptions(Context context, AudioClassifierOptions options) {
    OutputHandler<AudioClassifierResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<AudioClassifierResult, Void>() {
          @Override
          public AudioClassifierResult convertToTaskResult(List<Packet> packets) {
            try {
              if (!packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).isEmpty()) {
                // For audio stream mode.
                return AudioClassifierResult.createFromProto(
                    PacketGetter.getProto(
                        packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX),
                        ClassificationsProto.ClassificationResult.getDefaultInstance()),
                    packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).getTimestamp()
                        / MICROSECONDS_PER_MILLISECOND);
              } else {
                // For audio clips mode.
                return AudioClassifierResult.createFromProtoList(
                    PacketGetter.getProtoVector(
                        packets.get(TIMESTAMPED_CLASSIFICATIONS_OUT_STREAM_INDEX),
                        ClassificationsProto.ClassificationResult.parser()),
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
      ResultListener<AudioClassifierResult, Void> resultListener =
          new ResultListener<AudioClassifierResult, Void>() {
            @Override
            public void run(AudioClassifierResult audioClassifierResult, Void input) {
              options.resultListener().get().run(audioClassifierResult);
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
            TaskInfo.<AudioClassifierOptions>builder()
                .setTaskName(AudioClassifier.class.getSimpleName())
                .setTaskRunningModeName(options.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new AudioClassifier(runner, options.runningMode());
  }

  /**
   * Constructor to initialize an {@link AudioClassifier} from a {@link TaskRunner} and {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe audio task {@link RunningMode}.
   */
  private AudioClassifier(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, AUDIO_IN_STREAM_NAME, SAMPLE_RATE_IN_STREAM_NAME);
  }

  /*
   * Performs audio classification on the provided audio clip. Only use this method when the
   * AudioClassifier is created with the audio clips mode.
   *
   * <p>The audio clip is represented as a MediaPipe {@link AudioData} object  The method accepts
   * audio clips with various length and audio sample rate.  It's required to provide the
   * corresponding audio sample rate within the {@link AudioData} object.
   *
   * <p>The input audio clip may be longer than what the model is able to process in a single
   * inference. When this occurs, the input audio clip is split into multiple chunks starting at
   * different timestamps. For this reason, this function returns a vector of ClassificationResult
   * objects, each associated with a timestamp corresponding to the start (in milliseconds) of the
   * chunk data that was classified, e.g:
   *
   * ClassificationResult #0 (first chunk of data):
   *  timestamp_ms: 0 (starts at 0ms)
   *  classifications #0 (single head model):
   *   category #0:
   *    category_name: "Speech"
   *    score: 0.6
   *   category #1:
   *    category_name: "Music"
   *    score: 0.2
   * ClassificationResult #1 (second chunk of data):
   *  timestamp_ms: 800 (starts at 800ms)
   *  classifications #0 (single head model):
   *   category #0:
   *    category_name: "Speech"
   *    score: 0.5
   *   category #1:
   *    category_name: "Silence"
   *    score: 0.1
   *
   * @param audioClip a MediaPipe {@link AudioData} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public AudioClassifierResult classify(AudioData audioClip) {
    return (AudioClassifierResult) processAudioClip(audioClip);
  }

  /*
   * Sends audio data (a block in a continuous audio stream) to perform audio classification, and
   * the results will be available via the {@link ResultListener} provided in the
   * {@link AudioClassifierOptions}. Only use this method when the AudioClassifier is created with
   * the audio stream mode.
   *
   * <p>The audio block is represented as a MediaPipe {@link AudioData} object. The audio data will
   * be resampled, accumulated, and framed to the proper size for the underlying model to consume.
   * It's required to provide the corresponding audio sample rate within {@link AudioData} object as
   * well as a timestamp (in milliseconds) to indicate the start time of the input audio block. The
   * timestamps must be monotonically increasing. This method will return immediately after
   * the input audio data is accepted. The results will be available in the `resultListener`
   * provided in the `AudioClassifierOptions`. The `classifyAsync` method is designed to process
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
  public void classifyAsync(AudioData audioBlock, long timestampMs) {
    checkOrSetSampleRate(audioBlock.getFormat().getSampleRate());
    sendAudioStreamData(audioBlock, timestampMs);
  }

  /** Options for setting up and {@link AudioClassifier}. */
  @AutoValue
  public abstract static class AudioClassifierOptions extends TaskOptions {

    /** Builder for {@link AudioClassifierOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the audio classifier task. */
      public abstract Builder setBaseOptions(BaseOptions baseOptions);

      /**
       * Sets the {@link RunningMode} for the audio classifier task. Default to the audio clips
       * mode. Image classifier has two modes:
       *
       * <ul>
       *   <li>AUDIO_CLIPS: The mode for running audio classification on audio clips. Users feed
       *       audio clips to the `classify` method, and will receive the classification results as
       *       the return value.
       *   <li>AUDIO_STREAM: The mode for running audio classification on the audio stream, such as
       *       from microphone. Users call `classifyAsync` to push the audio data into the
       *       AudioClassifier, the classification results will be available in the result callback
       *       when the audio classifier finishes the work.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode runningMode);

      /**
       * Sets the optional locale to use for display names specified through the TFLite Model
       * Metadata, if any.
       */
      public abstract Builder setDisplayNamesLocale(String locale);

      /**
       * Sets the optional maximum number of top-scored classification results to return.
       *
       * <p>If not set, all available results are returned. If set, must be > 0.
       */
      public abstract Builder setMaxResults(Integer maxResults);

      /**
       * Sets the optional score threshold. Results with score below this value are rejected.
       *
       * <p>Overrides the score threshold specified in the TFLite Model Metadata, if any.
       */
      public abstract Builder setScoreThreshold(Float scoreThreshold);

      /**
       * Sets the optional allowlist of category names.
       *
       * <p>If non-empty, detection results whose category name is not in this set will be filtered
       * out. Duplicate or unknown category names are ignored. Mutually exclusive with {@code
       * categoryDenylist}.
       */
      public abstract Builder setCategoryAllowlist(List<String> categoryAllowlist);

      /**
       * Sets the optional denylist of category names.
       *
       * <p>If non-empty, detection results whose category name is in this set will be filtered out.
       * Duplicate or unknown category names are ignored. Mutually exclusive with {@code
       * categoryAllowlist}.
       */
      public abstract Builder setCategoryDenylist(List<String> categoryDenylist);

      /**
       * Sets the {@link ResultListener} to receive the classification results asynchronously when
       * the audio classifier is in the audio stream mode.
       */
      public abstract Builder setResultListener(
          PureResultListener<AudioClassifierResult> resultListener);

      /** Sets an optional {@link ErrorListener}. */
      public abstract Builder setErrorListener(ErrorListener errorListener);

      abstract AudioClassifierOptions autoBuild();

      /**
       * Validates and builds the {@link AudioClassifierOptions} instance.
       *
       * @throws IllegalArgumentException if any of the set options are invalid.
       */
      public final AudioClassifierOptions build() {
        AudioClassifierOptions options = autoBuild();
        if (options.runningMode() == RunningMode.AUDIO_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The audio classifier is in the audio stream mode, a user-defined result listener"
                    + " must be provided in the AudioClassifierOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The audio classifier is in the audio clips mode, a user-defined result listener"
                  + " shouldn't be provided in AudioClassifierOptions.");
        }
        if (options.maxResults().isPresent() && options.maxResults().get() <= 0) {
          throw new IllegalArgumentException("If specified, maxResults must be > 0.");
        }
        if (!options.categoryAllowlist().isEmpty() && !options.categoryDenylist().isEmpty()) {
          throw new IllegalArgumentException(
              "Category allowlist and denylist are mutually exclusive.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<String> displayNamesLocale();

    abstract Optional<Integer> maxResults();

    abstract Optional<Float> scoreThreshold();

    abstract List<String> categoryAllowlist();

    abstract List<String> categoryDenylist();

    abstract Optional<PureResultListener<AudioClassifierResult>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_AudioClassifier_AudioClassifierOptions.Builder()
          .setRunningMode(RunningMode.AUDIO_CLIPS)
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList());
    }

    /**
     * Converts a {@link AudioClassifierOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() == RunningMode.AUDIO_STREAM);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      ClassifierOptionsProto.ClassifierOptions.Builder classifierOptionsBuilder =
          ClassifierOptionsProto.ClassifierOptions.newBuilder();
      displayNamesLocale().ifPresent(classifierOptionsBuilder::setDisplayNamesLocale);
      maxResults().ifPresent(classifierOptionsBuilder::setMaxResults);
      scoreThreshold().ifPresent(classifierOptionsBuilder::setScoreThreshold);
      if (!categoryAllowlist().isEmpty()) {
        classifierOptionsBuilder.addAllCategoryAllowlist(categoryAllowlist());
      }
      if (!categoryDenylist().isEmpty()) {
        classifierOptionsBuilder.addAllCategoryDenylist(categoryDenylist());
      }
      AudioClassifierGraphOptionsProto.AudioClassifierGraphOptions.Builder taskOptionsBuilder =
          AudioClassifierGraphOptionsProto.AudioClassifierGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setClassifierOptions(classifierOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              AudioClassifierGraphOptionsProto.AudioClassifierGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
