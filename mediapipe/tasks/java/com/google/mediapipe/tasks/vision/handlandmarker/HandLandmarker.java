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

package com.google.mediapipe.tasks.vision.handlandmarker;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Connection;
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
import com.google.mediapipe.tasks.vision.handdetector.proto.HandDetectorGraphOptionsProto;
import com.google.mediapipe.tasks.vision.handlandmarker.proto.HandLandmarkerGraphOptionsProto;
import com.google.mediapipe.tasks.vision.handlandmarker.proto.HandLandmarksDetectorGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Performs hand landmarks detection on images.
 *
 * <p>This API expects a pre-trained hand landmarks model asset bundle. See <TODO link
 * to the DevSite documentation page>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that hand landmarks detection runs on.
 *       </ul>
 *   <li>Output HandLandmarkerResult {@link HandLandmarkerResult}
 *       <ul>
 *         <li>A HandLandmarkerResult containing hand landmarks.
 *       </ul>
 * </ul>
 */
public final class HandLandmarker extends BaseVisionTaskApi {
  private static final String TAG = HandLandmarker.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "LANDMARKS:hand_landmarks",
              "WORLD_LANDMARKS:world_hand_landmarks",
              "HANDEDNESS:handedness",
              "IMAGE:image_out"));
  private static final int LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int WORLD_LANDMARKS_OUT_STREAM_INDEX = 1;
  private static final int HANDEDNESS_OUT_STREAM_INDEX = 2;
  private static final int IMAGE_OUT_STREAM_INDEX = 3;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link HandLandmarker} instance from a model file and the default {@link
   * HandLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the hand landmarks model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link HandLandmarker} creation.
   */
  public static HandLandmarker createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, HandLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link HandLandmarker} instance from a model file and the default {@link
   * HandLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the hand landmarks model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link HandLandmarker} creation.
   */
  public static HandLandmarker createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, HandLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link HandLandmarker} instance from a model buffer and the default {@link
   * HandLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link HandLandmarker} creation.
   */
  public static HandLandmarker createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, HandLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link HandLandmarker} instance from a {@link HandLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param landmarkerOptions a {@link HandLandmarkerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link HandLandmarker} creation.
   */
  public static HandLandmarker createFromOptions(
      Context context, HandLandmarkerOptions landmarkerOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<HandLandmarkerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<HandLandmarkerResult, MPImage>() {
          @Override
          public HandLandmarkerResult convertToTaskResult(List<Packet> packets) {
            // If there is no hands detected in the image, just returns empty lists.
            if (packets.get(LANDMARKS_OUT_STREAM_INDEX).isEmpty()) {
              return HandLandmarkerResult.create(
                  new ArrayList<>(),
                  new ArrayList<>(),
                  new ArrayList<>(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
            }
            return HandLandmarkerResult.create(
                PacketGetter.getProtoVector(
                    packets.get(LANDMARKS_OUT_STREAM_INDEX), NormalizedLandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(WORLD_LANDMARKS_OUT_STREAM_INDEX), LandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(HANDEDNESS_OUT_STREAM_INDEX), ClassificationList.parser()),
                BaseVisionTaskApi.generateResultTimestampMs(
                    landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    landmarkerOptions.resultListener().ifPresent(handler::setResultListener);
    landmarkerOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<HandLandmarkerOptions>builder()
                .setTaskName(HandLandmarker.class.getSimpleName())
                .setTaskRunningModeName(landmarkerOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(landmarkerOptions)
                .setEnableFlowLimiting(landmarkerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new HandLandmarker(runner, landmarkerOptions.runningMode());
  }

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_PALM_CONNECTIONS =
      HandLandmarksConnections.HAND_PALM_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_THUMB_CONNECTIONS =
      HandLandmarksConnections.HAND_THUMB_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_INDEX_FINGER_CONNECTIONS =
      HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_MIDDLE_FINGER_CONNECTIONS =
      HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_RING_FINGER_CONNECTIONS =
      HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_PINKY_FINGER_CONNECTIONS =
      HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS;

  /**
   * Constructor to initialize an {@link HandLandmarker} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private HandLandmarker(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs hand landmarks detection on the provided single image with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * HandLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public HandLandmarkerResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs hand landmarks detection on the provided single image. Only use this method when the
   * {@link HandLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java
   * doc for input image format.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
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
  public HandLandmarkerResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (HandLandmarkerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs hand landmarks detection on the provided video frame with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * HandLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public HandLandmarkerResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs hand landmarks detection on the provided video frame. Only use this method when the
   * {@link HandLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
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
  public HandLandmarkerResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (HandLandmarkerResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform hand landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link HandLandmarkerOptions}. Only use this method when the
   * {@link HandLandmarker } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the hand landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
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
   * Sends live image data to perform hand landmarks detection, and the results will be available
   * via the {@link ResultListener} provided in the {@link HandLandmarkerOptions}. Only use this
   * method when the {@link HandLandmarker} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the hand landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link HandLandmarker} supports the following color space types:
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

  /** Options for setting up an {@link HandLandmarker}. */
  @AutoValue
  public abstract static class HandLandmarkerOptions extends TaskOptions {

    /** Builder for {@link HandLandmarkerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the hand landmarker task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the hand landmarker task. Default to the image mode. Hand
       * landmarker has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting hand landmarks on single image inputs.
       *   <li>VIDEO: The mode for detecting hand landmarks on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting hand landmarks on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /** Sets the maximum number of hands can be detected by the HandLandmarker. */
      public abstract Builder setNumHands(Integer value);

      /** Sets minimum confidence score for the hand detection to be considered successful */
      public abstract Builder setMinHandDetectionConfidence(Float value);

      /** Sets minimum confidence score of hand presence score in the hand landmark detection. */
      public abstract Builder setMinHandPresenceConfidence(Float value);

      /** Sets the minimum confidence score for the hand tracking to be considered successful. */
      public abstract Builder setMinTrackingConfidence(Float value);

      /**
       * Sets the result listener to receive the detection results asynchronously when the hand
       * landmarker is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<HandLandmarkerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract HandLandmarkerOptions autoBuild();

      /**
       * Validates and builds the {@link HandLandmarkerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the hand landmarker is
       *     in the live stream mode.
       */
      public final HandLandmarkerOptions build() {
        HandLandmarkerOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The hand landmarker is in the live stream mode, a user-defined result listener"
                    + " must be provided in HandLandmarkerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The hand landmarker is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in HandLandmarkerOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<Integer> numHands();

    abstract Optional<Float> minHandDetectionConfidence();

    abstract Optional<Float> minHandPresenceConfidence();

    abstract Optional<Float> minTrackingConfidence();

    abstract Optional<ResultListener<HandLandmarkerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_HandLandmarker_HandLandmarkerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setNumHands(1)
          .setMinHandDetectionConfidence(0.5f)
          .setMinHandPresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f);
    }

    /** Converts a {@link HandLandmarkerOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      HandLandmarkerGraphOptionsProto.HandLandmarkerGraphOptions.Builder taskOptionsBuilder =
          HandLandmarkerGraphOptionsProto.HandLandmarkerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      // Setup HandDetectorGraphOptions.
      HandDetectorGraphOptionsProto.HandDetectorGraphOptions.Builder
          handDetectorGraphOptionsBuilder =
              HandDetectorGraphOptionsProto.HandDetectorGraphOptions.newBuilder();
      numHands().ifPresent(handDetectorGraphOptionsBuilder::setNumHands);
      minHandDetectionConfidence()
          .ifPresent(handDetectorGraphOptionsBuilder::setMinDetectionConfidence);

      // Setup HandLandmarkerGraphOptions.
      HandLandmarksDetectorGraphOptionsProto.HandLandmarksDetectorGraphOptions.Builder
          handLandmarksDetectorGraphOptionsBuilder =
              HandLandmarksDetectorGraphOptionsProto.HandLandmarksDetectorGraphOptions.newBuilder();
      minHandPresenceConfidence()
          .ifPresent(handLandmarksDetectorGraphOptionsBuilder::setMinDetectionConfidence);
      minTrackingConfidence().ifPresent(taskOptionsBuilder::setMinTrackingConfidence);

      taskOptionsBuilder
          .setHandDetectorGraphOptions(handDetectorGraphOptionsBuilder.build())
          .setHandLandmarksDetectorGraphOptions(handLandmarksDetectorGraphOptionsBuilder.build());

      return CalculatorOptions.newBuilder()
          .setExtension(
              HandLandmarkerGraphOptionsProto.HandLandmarkerGraphOptions.ext,
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
      throw new IllegalArgumentException("HandLandmarker doesn't support region-of-interest.");
    }
  }
}
