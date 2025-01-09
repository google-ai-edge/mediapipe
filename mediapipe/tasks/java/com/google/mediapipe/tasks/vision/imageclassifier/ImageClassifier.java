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

package com.google.mediapipe.tasks.vision.imageclassifier;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.ClassificationResult;
import com.google.mediapipe.tasks.components.containers.proto.ClassificationsProto;
import com.google.mediapipe.tasks.components.processors.proto.ClassifierOptionsProto;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.OutputHandler.ResultListener;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageclassifier.proto.ImageClassifierGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs classification on images.
 *
 * <p>The API expects a TFLite model with optional, but strongly recommended, <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <p>The API supports models with one image input tensor and one or more output tensors. To be more
 * specific, here are the requirements.
 *
 * <ul>
 *   <li>Input image tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32})
 *       <ul>
 *         <li>image input of size {@code [batch x height x width x channels]}.
 *         <li>batch inference is not supported ({@code batch} is required to be 1).
 *         <li>only RGB inputs are supported ({@code channels} is required to be 3).
 *         <li>if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the
 *             metadata for input normalization.
 *       </ul>
 *   <li>At least one output tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32}) with:
 *       <ul>
 *         <li>{@code N} classes and either 2 or 4 dimensions, i.e. {@code [1 x N]} or {@code [1 x 1
 *             x 1 x N]}
 *         <li>optional (but recommended) label map(s) as AssociatedFile-s with type
 *             TENSOR_AXIS_LABELS, containing one label per line. The first such AssociatedFile (if
 *             any) is used to fill the {@code class_name} field of the results. The {@code
 *             display_name} field is filled from the AssociatedFile (if any) whose locale matches
 *             the {@code display_names_locale} field of the {@code ImageClassifierOptions} used at
 *             creation time ("en" by default, i.e. English). If none of these are available, only
 *             the {@code index} field of the results will be filled.
 *         <li>optional score calibration can be attached using ScoreCalibrationOptions and an
 *             AssociatedFile with type TENSOR_AXIS_SCORE_CALIBRATION. See <a
 *             href="https://github.com/google/mediapipe/blob/master/mediapipe/tasks/metadata/metadata_schema.fbs">
 *             metadata_schema.fbs</a> for more details.
 *       </ul>
 * </ul>
 *
 * <p>An example of such model can be found <a
 * href="https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1">
 * TensorFlow Hub</a>.
 */
public final class ImageClassifier extends BaseVisionTaskApi {
  private static final String TAG = ImageClassifier.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("CLASSIFICATIONS:classifications_out", "IMAGE:image_out"));
  private static final int CLASSIFICATIONS_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
    ProtoUtil.registerTypeName(
        ClassificationsProto.ClassificationResult.class,
        "mediapipe.tasks.components.containers.proto.ClassificationResult");
  }

  /**
   * Creates an {@link ImageClassifier} instance from a model file and default {@link
   * ImageClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the classification model in the assets.
   * @throws MediaPipeException if there is an error during {@link ImageClassifier} creation.
   */
  public static ImageClassifier createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, ImageClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ImageClassifier} instance from a model file and default {@link
   * ImageClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the classification model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link ImageClassifier} creation.
   */
  public static ImageClassifier createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, ImageClassifierOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates an {@link ImageClassifier} instance from a model buffer and default {@link
   * ImageClassifierOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the
   *     classification model.
   * @throws MediaPipeException if there is an error during {@link ImageClassifier} creation.
   */
  public static ImageClassifier createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, ImageClassifierOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ImageClassifier} instance from an {@link ImageClassifierOptions} instance.
   *
   * @param context an Android {@link Context}.
   * @param options an {@link ImageClassifierOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageClassifier} creation.
   */
  public static ImageClassifier createFromOptions(Context context, ImageClassifierOptions options) {
    OutputHandler<ImageClassifierResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageClassifierResult, MPImage>() {
          @Override
          public ImageClassifierResult convertToTaskResult(List<Packet> packets) {
            try {
              return ImageClassifierResult.create(
                  ClassificationResult.createFromProto(
                      PacketGetter.getProto(
                          packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX),
                          ClassificationsProto.ClassificationResult.getDefaultInstance())),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      options.runningMode(), packets.get(CLASSIFICATIONS_OUT_STREAM_INDEX)));
            } catch (IOException e) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(), e.getMessage());
            }
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    options.resultListener().ifPresent(handler::setResultListener);
    options.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ImageClassifierOptions>builder()
                .setTaskName(ImageClassifier.class.getSimpleName())
                .setTaskRunningModeName(options.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(options)
                .setEnableFlowLimiting(options.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new ImageClassifier(runner, options.runningMode());
  }

  /**
   * Constructor to initialize an {@link ImageClassifier} from a {@link TaskRunner} and {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private ImageClassifier(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs classification on the provided single image with default image processing options,
   * i.e. using the whole image as region-of-interest and without any rotation applied. Only use
   * this method when the {@link ImageClassifier} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageClassifierResult classify(MPImage image) {
    return classify(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs classification on the provided single image. Only use this method when the {@link
   * ImageClassifier} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageClassifierResult classify(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    return (ImageClassifierResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs classification on the provided video frame with default image processing options, i.e.
   * using the whole image as region-of-interest and without any rotation applied. Only use this
   * method when the {@link ImageClassifier} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageClassifierResult classifyForVideo(MPImage image, long timestampMs) {
    return classifyForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs classification on the provided video frame. Only use this method when the {@link
   * ImageClassifier} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageClassifierResult classifyForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    return (ImageClassifierResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform classification with default image processing options, i.e.
   * using the whole image as region-of-interest and without any rotation applied, and the results
   * will be available via the {@link ResultListener} provided in the {@link
   * ImageClassifierOptions}. Only use this method when the {@link ImageClassifier} is created with
   * {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void classifyAsync(MPImage image, long timestampMs) {
    classifyAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform classification, and the results will be available via the
   * {@link ResultListener} provided in the {@link ImageClassifierOptions}. Only use this method
   * when the {@link ImageClassifier} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageClassifier} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void classifyAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  /** Options for setting up and {@link ImageClassifier}. */
  @AutoValue
  public abstract static class ImageClassifierOptions extends TaskOptions {

    /** Builder for {@link ImageClassifierOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the image classifier task. */
      public abstract Builder setBaseOptions(BaseOptions baseOptions);

      /**
       * Sets the {@link RunningMode} for the image classifier task. Default to the image mode.
       * Image classifier has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for performing classification on single image inputs.
       *   <li>VIDEO: The mode for performing classification on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for performing classification on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the classification results asynchronously.
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
       * the image classifier is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<ImageClassifierResult, MPImage> resultListener);

      /** Sets an optional {@link ErrorListener}. */
      public abstract Builder setErrorListener(ErrorListener errorListener);

      abstract ImageClassifierOptions autoBuild();

      /**
       * Validates and builds the {@link ImageClassifierOptions} instance. *
       *
       * @throws IllegalArgumentException if any of the set options are invalid.
       */
      public final ImageClassifierOptions build() {
        ImageClassifierOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The image classifier is in the live stream mode, a user-defined result listener"
                    + " must be provided in the ImageClassifierOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The image classifier is in the image or video mode, a user-defined result listener"
                  + " shouldn't be provided in ImageClassifierOptions.");
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

    abstract Optional<ResultListener<ImageClassifierResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_ImageClassifier_ImageClassifierOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList());
    }

    /**
     * Converts a {@link ImageClassifierOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() != RunningMode.IMAGE);
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
      ImageClassifierGraphOptionsProto.ImageClassifierGraphOptions.Builder taskOptionsBuilder =
          ImageClassifierGraphOptionsProto.ImageClassifierGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setClassifierOptions(classifierOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              ImageClassifierGraphOptionsProto.ImageClassifierGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
