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

package com.google.mediapipe.tasks.vision.objectdetector;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
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
import com.google.mediapipe.tasks.vision.objectdetector.proto.ObjectDetectorOptionsProto;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs object detection on images.
 *
 * <p>The API expects a TFLite model with <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <p>The API supports models with one image input tensor and four output tensors. To be more
 * specific, here are the requirements.
 *
 * <ul>
 *   <li>Input image tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32})
 *       <ul>
 *         <li>image input of size {@code [batch x height x width x channels]}.
 *         <li>batch inference is not supported ({@code batch} is required to be 1).
 *         <li>only RGB inputs are supported ({@code channels} is required to be 3).
 *         <li>if type is {@code kTfLiteFloat32}, NormalizationOptions are required to be attached
 *             to the metadata for input normalization.
 *       </ul>
 *   <li>Output tensors must be the 4 outputs of a {@code DetectionPostProcess} op, i.e:
 *       <ul>
 *         <li>Location tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results x 4]}, the inner array representing
 *                   bounding boxes in the form [top, left, right, bottom].
 *               <li>{@code BoundingBoxProperties} are required to be attached to the metadata and
 *                   must specify {@code type=BOUNDARIES} and {@code coordinate_type=RATIO}.
 *             </ul>
 *         <li>Classes tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results]}, each value representing the integer
 *                   index of a class.
 *               <li>if label maps are attached to the metadata as {@code TENSOR_VALUE_LABELS}
 *                   associated files, they are used to convert the tensor values into labels.
 *             </ul>
 *         <li>scores tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results]}, each value representing the score of
 *                   the detected object.
 *             </ul>
 *         <li>Number of detection tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>integer num_results as a tensor of size {@code [1]}.
 *             </ul>
 *       </ul>
 * </ul>
 *
 * <p>An example of such model can be found on <a
 * href="https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/metadata/1">TensorFlow
 * Hub.</a>.
 */
public final class ObjectDetector extends BaseVisionTaskApi {
  private static final String TAG = ObjectDetector.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("DETECTIONS:detections_out", "IMAGE:image_out"));

  private static final int DETECTIONS_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;
  private static final String TASK_GRAPH_NAME = "mediapipe.tasks.vision.ObjectDetectorGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates an {@link ObjectDetector} instance from a model file and the default {@link
   * ObjectDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the detection model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link ObjectDetector} creation.
   */
  public static ObjectDetector createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, ObjectDetectorOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ObjectDetector} instance from a model file and the default {@link
   * ObjectDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the detection model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link ObjectDetector} creation.
   */
  public static ObjectDetector createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, ObjectDetectorOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates an {@link ObjectDetector} instance from a model buffer and the default {@link
   * ObjectDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link ObjectDetector} creation.
   */
  public static ObjectDetector createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, ObjectDetectorOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ObjectDetector} instance from an {@link ObjectDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param detectorOptions an {@link ObjectDetectorOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ObjectDetector} creation.
   */
  public static ObjectDetector createFromOptions(
      Context context, ObjectDetectorOptions detectorOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<ObjectDetectorResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ObjectDetectorResult, MPImage>() {
          @Override
          public ObjectDetectorResult convertToTaskResult(List<Packet> packets) {
            // If there is no object detected in the image, just returns empty lists.
            if (packets.get(DETECTIONS_OUT_STREAM_INDEX).isEmpty()) {
              return ObjectDetectorResult.create(
                  new ArrayList<>(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      detectorOptions.runningMode(), packets.get(DETECTIONS_OUT_STREAM_INDEX)));
            }
            return ObjectDetectorResult.create(
                PacketGetter.getProtoVector(
                    packets.get(DETECTIONS_OUT_STREAM_INDEX), Detection.parser()),
                BaseVisionTaskApi.generateResultTimestampMs(
                    detectorOptions.runningMode(), packets.get(DETECTIONS_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    detectorOptions.resultListener().ifPresent(handler::setResultListener);
    detectorOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ObjectDetectorOptions>builder()
                .setTaskName(ObjectDetector.class.getSimpleName())
                .setTaskRunningModeName(detectorOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(detectorOptions)
                .setEnableFlowLimiting(detectorOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new ObjectDetector(runner, detectorOptions.runningMode());
  }

  /**
   * Constructor to initialize an {@link ObjectDetector} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private ObjectDetector(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs object detection on the provided single image with default image processing options,
   * i.e. without any rotation applied. Only use this method when the {@link ObjectDetector} is
   * created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public ObjectDetectorResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs object detection on the provided single image. Only use this method when the {@link
   * ObjectDetector} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public ObjectDetectorResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (ObjectDetectorResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs object detection on the provided video frame with default image processing options,
   * i.e. without any rotation applied. Only use this method when the {@link ObjectDetector} is
   * created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public ObjectDetectorResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs object detection on the provided video frame. Only use this method when the {@link
   * ObjectDetector} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public ObjectDetectorResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (ObjectDetectorResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform object detection with default image processing options, i.e.
   * without any rotation applied, and the results will be available via the {@link ResultListener}
   * provided in the {@link ObjectDetectorOptions}. Only use this method when the {@link
   * ObjectDetector} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void detectAsync(MPImage image, long timestampMs) {
    detectAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform object detection, and the results will be available via the
   * {@link ResultListener} provided in the {@link ObjectDetectorOptions}. Only use this method when
   * the {@link ObjectDetector} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ObjectDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error.
   */
  public void detectAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  /** Options for setting up an {@link ObjectDetector}. */
  @AutoValue
  public abstract static class ObjectDetectorOptions extends TaskOptions {

    /** Builder for {@link ObjectDetectorOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the object detector task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the {@link RunningMode} for the object detector task. Default to the image mode.
       * Object detector has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting objects on single image inputs.
       *   <li>VIDEO: The mode for detecting objects on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting objects on a live stream of input data, such
       *       as from camera. In this mode, {@code setResultListener} must be called to set up a
       *       listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /**
       * Sets the optional locale to use for display names specified through the TFLite Model
       * Metadata, if any.
       */
      public abstract Builder setDisplayNamesLocale(String value);

      /**
       * Sets the optional maximum number of top-scored detection results to return.
       *
       * <p>Overrides the ones provided in the model metadata. Results below this value are
       * rejected.
       */
      public abstract Builder setMaxResults(Integer value);

      /**
       * Sets the optional score threshold that overrides the one provided in the model metadata (if
       * any). Results below this value are rejected.
       */
      public abstract Builder setScoreThreshold(Float value);

      /**
       * Sets the optional allowlist of category names.
       *
       * <p>If non-empty, detection results whose category name is not in this set will be filtered
       * out. Duplicate or unknown category names are ignored. Mutually exclusive with {@code
       * categoryDenylist}.
       */
      public abstract Builder setCategoryAllowlist(List<String> value);

      /**
       * Sets the optional denylist of category names.
       *
       * <p>If non-empty, detection results whose category name is in this set will be filtered out.
       * Duplicate or unknown category names are ignored. Mutually exclusive with {@code
       * categoryAllowlist}.
       */
      public abstract Builder setCategoryDenylist(List<String> value);

      /**
       * Sets the {@link ResultListener} to receive the detection results asynchronously when the
       * object detector is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<ObjectDetectorResult, MPImage> value);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract ObjectDetectorOptions autoBuild();

      /**
       * Validates and builds the {@link ObjectDetectorOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the object detector is
       *     in the live stream mode.
       */
      public final ObjectDetectorOptions build() {
        ObjectDetectorOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The object detector is in the live stream mode, a user-defined result listener"
                    + " must be provided in ObjectDetectorOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The object detector is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in ObjectDetectorOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<String> displayNamesLocale();

    abstract Optional<Integer> maxResults();

    abstract Optional<Float> scoreThreshold();

    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryAllowlist();

    @SuppressWarnings("AutoValueImmutableFields")
    abstract List<String> categoryDenylist();

    abstract Optional<ResultListener<ObjectDetectorResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_ObjectDetector_ObjectDetectorOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setCategoryAllowlist(Collections.emptyList())
          .setCategoryDenylist(Collections.emptyList());
    }

    /** Converts a {@link ObjectDetectorOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() != RunningMode.IMAGE);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      ObjectDetectorOptionsProto.ObjectDetectorOptions.Builder taskOptionsBuilder =
          ObjectDetectorOptionsProto.ObjectDetectorOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder);
      displayNamesLocale().ifPresent(taskOptionsBuilder::setDisplayNamesLocale);
      maxResults().ifPresent(taskOptionsBuilder::setMaxResults);
      scoreThreshold().ifPresent(taskOptionsBuilder::setScoreThreshold);
      if (!categoryAllowlist().isEmpty()) {
        taskOptionsBuilder.addAllCategoryAllowlist(categoryAllowlist());
      }
      if (!categoryDenylist().isEmpty()) {
        taskOptionsBuilder.addAllCategoryDenylist(categoryDenylist());
      }
      return CalculatorOptions.newBuilder()
          .setExtension(
              ObjectDetectorOptionsProto.ObjectDetectorOptions.ext, taskOptionsBuilder.build())
          .build();
    }
  }

  /**
   * Validates that the provided {@link ImageProcessingOptions} doesn't contain a
   * region-of-interest.
   */
  private static void validateImageProcessingOptions(
      ImageProcessingOptions imageProcessingOptions) {
    if (imageProcessingOptions.regionOfInterest().isPresent()) {
      throw new IllegalArgumentException("ObjectDetector doesn't support region-of-interest.");
    }
  }
}
