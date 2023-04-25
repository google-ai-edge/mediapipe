// Copyright 2023 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.text.languagedetector;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.ClassificationResult;
import com.google.mediapipe.tasks.components.containers.Classifications;
import com.google.mediapipe.tasks.components.containers.proto.ClassificationsProto;
import com.google.mediapipe.tasks.components.processors.proto.ClassifierOptionsProto;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.text.textclassifier.proto.TextClassifierGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Predicts the language of an input text.
 *
 * <p>This API expects a TFLite model with <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata</a> that contains
 * the mandatory (described below) input tensors, output tensor, and the language codes in an
 * AssociatedFile.
 *
 * <ul>
 *   <li>Input tensor
 *       <ul>
 *         <li>One input tensor ({@code kTfLiteString}) of shape [1] containing the input string.
 *       </ul>
 *   <li>Output tensor
 *       <ul>
 *         <li>One output tensor ({@code kTfLiteFloat32}) of shape {@code [1 x N]} where {@code N}
 *             is the number of languages.
 *       </ul>
 * </ul>
 */
public final class LanguageDetector implements AutoCloseable {
  private static final String TAG = LanguageDetector.class.getSimpleName();
  private static final String TEXT_IN_STREAM_NAME = "text_in";

  private static final List<String> inputStreams =
      Collections.unmodifiableList(Arrays.asList("TEXT:" + TEXT_IN_STREAM_NAME));

  private static final List<String> outputStreams =
      Collections.unmodifiableList(Arrays.asList("CLASSIFICATIONS:classifications_out"));

  private static final int CLASSIFICATIONS_OUT_STREAM_INDEX = 0;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.text.text_classifier.TextClassifierGraph";
  private final TaskRunner runner;

  static {
    System.loadLibrary("mediapipe_tasks_text_jni");
    ProtoUtil.registerTypeName(
        ClassificationsProto.ClassificationResult.class,
        "mediapipe.tasks.components.containers.proto.ClassificationResult");
  }

  /**
   * Creates a {@link LanguageDetector} instance from a model file and the default {@link
   * LanguageDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the text model with metadata in the assets.
   * @throws MediaPipeException if there is is an error during {@link LanguageDetector} creation.
   */
  public static LanguageDetector createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, LanguageDetectorOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link LanguageDetector} instance from a model file and the default {@link
   * LanguageDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the text model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link LanguageDetector} creation.
   */
  public static LanguageDetector createFromFile(Context context, File modelFile)
      throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, LanguageDetectorOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link LanguageDetector} instance from {@link LanguageDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param options a {@link LanguageDetectorOptions} instance.
   * @throws MediaPipeException if there is an error during {@link LanguageDetector} creation.
   */
  public static LanguageDetector createFromOptions(
      Context context, LanguageDetectorOptions options) {
    OutputHandler<LanguageDetectorResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<LanguageDetectorResult, Void>() {
          @Override
          public LanguageDetectorResult convertToTaskResult(List<Packet> packets) {
            if (packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).isEmpty()) {
              List<Category> emptyCategories = new ArrayList<>();
              Classifications emptyClassifications =
                  Classifications.create(emptyCategories, 0, Optional.empty());
              ClassificationResult classificationResult =
                  ClassificationResult.create(
                      Arrays.asList(emptyClassifications), Optional.empty());
              return LanguageDetectorResult.create(
                  classificationResult,
                  packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).getTimestamp());
            }
            try {
              return LanguageDetectorResult.create(
                  ClassificationResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX),
                          ClassificationsProto.ClassificationResult.getDefaultInstance())),
                  packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).getTimestamp());
            } catch (IOException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.IO_EXCEPTION.ordinal(), e.getMessage());
            } catch (IllegalArgumentException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INVALID_ARGUMENT.ordinal(), e.getMessage());
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
            TaskInfo.<LanguageDetectorOptions>builder()
                .setTaskName(LanguageDetector.class.getSimpleName())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(inputStreams)
                .setOutputStreams(outputStreams)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new LanguageDetector(runner);
  }

  /**
   * Constructor to initialize a {@link LanguageDetector} from a {@link TaskRunner}.
   *
   * @param runner a {@link TaskRunner}.
   */
  private LanguageDetector(TaskRunner runner) {
    this.runner = runner;
  }

  /**
   * Predicts the language of the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public LanguageDetectorResult detect(String inputText) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(TEXT_IN_STREAM_NAME, runner.getPacketCreator().createString(inputText));
    return (LanguageDetectorResult) runner.process(inputPackets);
  }

  /** Closes and cleans up the {@link LanguageDetector}. */
  @Override
  public void close() {
    runner.close();
  }

  /** Options for setting up a {@link LanguageDetector}. */
  @AutoValue
  public abstract static class LanguageDetectorOptions extends TaskOptions {

    /** Builder for {@link LanguageDetectorOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the text classifier task. */
      public abstract Builder setBaseOptions(BaseOptions value);

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

      abstract LanguageDetectorOptions autoBuild();

      /**
       * Validates and builds the {@link LanguageDetectorOptions} instance.
       *
       * @throws IllegalArgumentException if any of the set options are invalid.
       */
      public final LanguageDetectorOptions build() {
        LanguageDetectorOptions options = autoBuild();
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

    abstract Optional<String> displayNamesLocale();

    abstract Optional<Integer> maxResults();

    abstract Optional<Float> scoreThreshold();

    // For backwards-compatibility reasons in OSS we want to avoid dependencies on libraries like
    // Guava, so we don't use ImmutableList.
    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryAllowlist();

    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryDenylist();

    @SuppressWarnings("AutoValueImmutableFields")
    public static Builder builder() {
      return new AutoValue_LanguageDetector_LanguageDetectorOptions.Builder()
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList());
    }

    /**
     * Converts a {@link LanguageDetectorOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
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
      TextClassifierGraphOptionsProto.TextClassifierGraphOptions taskOptions =
          TextClassifierGraphOptionsProto.TextClassifierGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setClassifierOptions(classifierOptionsBuilder)
              .build();
      return CalculatorOptions.newBuilder()
          .setExtension(TextClassifierGraphOptionsProto.TextClassifierGraphOptions.ext, taskOptions)
          .build();
    }
  }
}
