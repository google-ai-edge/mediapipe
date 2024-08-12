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

package com.google.mediapipe.tasks.vision.facedetector;

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
import com.google.mediapipe.tasks.vision.facedetector.proto.FaceDetectorGraphOptionsProto;
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
 * Performs face detection on images.
 *
 * <p>The API expects a TFLite model with <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that the face detector runs on.
 *       </ul>
 *   <li>Output FaceDetectorResult {@link FaceDetectorResult}
 *       <ul>
 *         <li>A FaceDetectorResult containing detected faces.
 *       </ul>
 * </ul>
 */
public final class FaceDetector extends BaseVisionTaskApi {
  private static final String TAG = FaceDetector.class.getSimpleName();
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
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.face_detector.FaceDetectorGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link FaceDetector} instance from a model file and the default {@link
   * FaceDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the detection model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link FaceDetector} creation.
   */
  public static FaceDetector createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, FaceDetectorOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link FaceDetector} instance from a model file and the default {@link
   * FaceDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the detection model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link FaceDetector} creation.
   */
  public static FaceDetector createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, FaceDetectorOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link FaceDetector} instance from a model buffer and the default {@link
   * FaceDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link FaceDetector} creation.
   */
  public static FaceDetector createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, FaceDetectorOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link FaceDetector} instance from a {@link FaceDetectorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param detectorOptions a {@link FaceDetectorOptions} instance.
   * @throws MediaPipeException if there is an error during {@link FaceDetector} creation.
   */
  public static FaceDetector createFromOptions(
      Context context, FaceDetectorOptions detectorOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<FaceDetectorResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<FaceDetectorResult, MPImage>() {
          @Override
          public FaceDetectorResult convertToTaskResult(List<Packet> packets) {
            // If there is no faces detected in the image, just returns empty lists.
            if (packets.get(DETECTIONS_OUT_STREAM_INDEX).isEmpty()) {
              return FaceDetectorResult.create(
                  new ArrayList<>(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      detectorOptions.runningMode(), packets.get(DETECTIONS_OUT_STREAM_INDEX)));
            }
            return FaceDetectorResult.create(
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
            TaskInfo.<FaceDetectorOptions>builder()
                .setTaskName(FaceDetector.class.getSimpleName())
                .setTaskRunningModeName(detectorOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(detectorOptions)
                .setEnableFlowLimiting(detectorOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new FaceDetector(runner, detectorOptions.runningMode());
  }

  /**
   * Constructor to initialize a {@link FaceDetector} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private FaceDetector(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs face detection on the provided single image with default image processing options,
   * i.e. without any rotation applied. Only use this method when the {@link FaceDetector} is
   * created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link FaceDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public FaceDetectorResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs face detection on the provided single image. Only use this method when the {@link
   * FaceDetector} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link FaceDetector} supports the following color space types:
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
  public FaceDetectorResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (FaceDetectorResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs face detection on the provided video frame with default image processing options, i.e.
   * without any rotation applied. Only use this method when the {@link FaceDetector} is created
   * with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link FaceDetector} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public FaceDetectorResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs face detection on the provided video frame. Only use this method when the {@link
   * FaceDetector} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link FaceDetector} supports the following color space types:
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
  public FaceDetectorResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (FaceDetectorResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform face detection with default image processing options, i.e.
   * without any rotation applied, and the results will be available via the {@link ResultListener}
   * provided in the {@link FaceDetectorOptions}. Only use this method when the {@link FaceDetector}
   * is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the face detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link FaceDetector} supports the following color space types:
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
   * Sends live image data to perform face detection, and the results will be available via the
   * {@link ResultListener} provided in the {@link FaceDetectorOptions}. Only use this method when
   * the {@link FaceDetector} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the face detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link FaceDetector} supports the following color space types:
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

  /** Options for setting up a {@link FaceDetector}. */
  @AutoValue
  public abstract static class FaceDetectorOptions extends TaskOptions {

    /** Builder for {@link FaceDetectorOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the face detector task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the {@link RunningMode} for the face detector task. Default to the image mode. face
       * detector has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting faces on single image inputs.
       *   <li>VIDEO: The mode for detecting faces on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting faces on a live stream of input data, such as
       *       from camera. In this mode, {@code setResultListener} must be called to set up a
       *       listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /**
       * Sets the minimum confidence score for the face detection to be considered successful. The
       * default minDetectionConfidence is 0.5.
       */
      public abstract Builder setMinDetectionConfidence(float value);

      /**
       * Sets the minimum non-maximum-suppression threshold for face detection to be considered
       * overlapped. The default minSuppressionThreshold is 0.3.
       */
      public abstract Builder setMinSuppressionThreshold(float value);

      /**
       * Sets the {@link ResultListener} to receive the detection results asynchronously when the
       * face detector is in the live stream mode.
       */
      public abstract Builder setResultListener(ResultListener<FaceDetectorResult, MPImage> value);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract FaceDetectorOptions autoBuild();

      /**
       * Validates and builds the {@link FaceDetectorOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the face detector is
       *     in the live stream mode.
       */
      public final FaceDetectorOptions build() {
        FaceDetectorOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The face detector is in the live stream mode, a user-defined result listener"
                    + " must be provided in FaceDetectorOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The face detector is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in FaceDetectorOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract float minDetectionConfidence();

    abstract float minSuppressionThreshold();

    abstract Optional<ResultListener<FaceDetectorResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_FaceDetector_FaceDetectorOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setMinDetectionConfidence(0.5f)
          .setMinSuppressionThreshold(0.3f);
    }

    /** Converts a {@link FaceDetectorOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() != RunningMode.IMAGE);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions.Builder taskOptionsBuilder =
          FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder);
      taskOptionsBuilder.setMinDetectionConfidence(minDetectionConfidence());
      taskOptionsBuilder.setMinSuppressionThreshold(minSuppressionThreshold());
      return CalculatorOptions.newBuilder()
          .setExtension(
              FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions.ext,
              taskOptionsBuilder.build())
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
      throw new IllegalArgumentException("FaceDetector doesn't support region-of-interest.");
    }
  }
}
