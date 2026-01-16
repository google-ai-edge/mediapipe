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

package com.google.mediapipe.tasks.text.textclassifier;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.tasks.components.containers.ClassificationResult;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Performs classification on text.
 *
 * <p>This API expects a TFLite model with (optional) <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata</a> that contains
 * the mandatory (described below) input tensors, output tensor, and the optional (but recommended)
 * label items as AssociatedFiles with type TENSOR_AXIS_LABELS per output classification tensor.
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
 *   <li>At least one output tensor ({@code kTfLiteFloat32}/{@code kBool}) with:
 *       <ul>
 *         <li>{@code N} classes and shape {@code [1 x N]}
 *         <li>optional (but recommended) label map(s) as AssociatedFile-s with type
 *             TENSOR_AXIS_LABELS, containing one label per line. The first such AssociatedFile (if
 *             any) is used to fill the {@code class_name} field of the results. The {@code
 *             display_name} field is filled from the AssociatedFile (if any) whose locale matches
 *             the {@code display_names_locale} field of the {@code TextClassifierOptions} used at
 *             creation time ("en" by default, i.e. English). If none of these are available, only
 *             the {@code index} field of the results will be filled.
 *       </ul>
 * </ul>
 */
public final class TextClassifier implements AutoCloseable {
  private static final String TAG = TextClassifier.class.getSimpleName();
  private static final String TEXT_IN_STREAM_NAME = "text_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("TEXT:" + TEXT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
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
   * Creates a {@link TextClassifier} instance from a model file and the default {@link
   * TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the text model with metadata in the assets.
   * @throws MediaPipeException if there is is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, TextClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link TextClassifier} instance from a model file and the default {@link
   * TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the text model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, TextClassifierOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link TextClassifier} instance from {@link TextClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param options a {@link TextClassifierOptions} instance.
   * @throws MediaPipeException if there is an error during {@link TextClassifier} creation.
   */
  public static TextClassifier createFromOptions(Context context, TextClassifierOptions options) {
    OutputHandler<TextClassifierResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<TextClassifierResult, Void>() {
          @Override
          public TextClassifierResult convertToTaskResult(List<Packet> packets) {
            try {
              return TextClassifierResult.create(
                  ClassificationResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX),
                          ClassificationsProto.ClassificationResult.getDefaultInstance())),
                  packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX).getTimestamp());
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
            TaskInfo.<TextClassifierOptions>builder()
                .setTaskName(TextClassifier.class.getSimpleName())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new TextClassifier(runner);
  }

  /**
   * Constructor to initialize a {@link TextClassifier} from a {@link TaskRunner}.
   *
   * @param runner a {@link TaskRunner}.
   */
  private TextClassifier(TaskRunner runner) {
    this.runner = runner;
  }

  /**
   * Performs classification on the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public TextClassifierResult classify(String inputText) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(TEXT_IN_STREAM_NAME, runner.getPacketCreator().createString(inputText));
    return (TextClassifierResult) runner.process(inputPackets);
  }

  /** Closes and cleans up the {@link TextClassifier}. */
  @Override
  public void close() {
    runner.close();
  }

  /** Options for setting up a {@link TextClassifier}. */
  @AutoValue
  public abstract static class TextClassifierOptions extends TaskOptions {

    /** Builder for {@link TextClassifierOptions}. */
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

      abstract TextClassifierOptions autoBuild();

      /**
       * Validates and builds the {@link TextClassifierOptions} instance.
       *
       * @throws IllegalArgumentException if any of the set options are invalid.
       */
      public final TextClassifierOptions build() {
        TextClassifierOptions options = autoBuild();
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

    abstract List<String> categoryAllowlist();

    abstract List<String> categoryDenylist();

    public static Builder builder() {
      return new AutoValue_TextClassifier_TextClassifierOptions.Builder()
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList());
    }

    /** Converts a {@link TextClassifierOptions} to a {@link CalculatorOptions} protobuf message. */
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
      TextClassifierGraphOptionsProto.TextClassifierGraphOptions.Builder taskOptionsBuilder =
          TextClassifierGraphOptionsProto.TextClassifierGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setClassifierOptions(classifierOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              TextClassifierGraphOptionsProto.TextClassifierGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
