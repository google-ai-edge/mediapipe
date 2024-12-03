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

package com.google.mediapipe.tasks.vision.gesturerecognizer;

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
import com.google.mediapipe.tasks.components.processors.ClassifierOptions;
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
import com.google.mediapipe.tasks.vision.gesturerecognizer.proto.GestureClassifierGraphOptionsProto;
import com.google.mediapipe.tasks.vision.gesturerecognizer.proto.GestureRecognizerGraphOptionsProto;
import com.google.mediapipe.tasks.vision.gesturerecognizer.proto.HandGestureRecognizerGraphOptionsProto;
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

/**
 * Performs gesture recognition on images.
 *
 * <p>This API expects a pre-trained hand gesture model asset bundle, or a custom one created using
 * Model Maker. See <TODO link to the DevSite documentation page>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that gesture recognition runs on.
 *       </ul>
 *   <li>Output GestureRecognizerResult {@link GestureRecognizerResult}
 *       <ul>
 *         <li>A GestureRecognizerResult containing hand landmarks and recognized hand gestures.
 *       </ul>
 * </ul>
 */
public final class GestureRecognizer extends BaseVisionTaskApi {
  private static final String TAG = GestureRecognizer.class.getSimpleName();
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
              "HAND_GESTURES:hand_gestures",
              "IMAGE:image_out"));
  private static final int LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int WORLD_LANDMARKS_OUT_STREAM_INDEX = 1;
  private static final int HANDEDNESS_OUT_STREAM_INDEX = 2;
  private static final int HAND_GESTURES_OUT_STREAM_INDEX = 3;
  private static final int IMAGE_OUT_STREAM_INDEX = 4;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.gesture_recognizer.GestureRecognizerGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link GestureRecognizer} instance from a model file and the default {@link
   * GestureRecognizerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the gesture recognition model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link GestureRecognizer} creation.
   */
  public static GestureRecognizer createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, GestureRecognizerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link GestureRecognizer} instance from a model file and the default {@link
   * GestureRecognizerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the gesture recognition model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link GestureRecognizer} creation.
   */
  public static GestureRecognizer createFromFile(Context context, File modelFile)
      throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, GestureRecognizerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link GestureRecognizer} instance from a model buffer and the default {@link
   * GestureRecognizerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link GestureRecognizer} creation.
   */
  public static GestureRecognizer createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, GestureRecognizerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link GestureRecognizer} instance from a {@link GestureRecognizerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param recognizerOptions a {@link GestureRecognizerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link GestureRecognizer} creation.
   */
  public static GestureRecognizer createFromOptions(
      Context context, GestureRecognizerOptions recognizerOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<GestureRecognizerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<GestureRecognizerResult, MPImage>() {
          @Override
          public GestureRecognizerResult convertToTaskResult(List<Packet> packets) {
            // If there is no hands detected in the image, just returns empty lists.
            if (packets.get(HAND_GESTURES_OUT_STREAM_INDEX).isEmpty()) {
              return GestureRecognizerResult.create(
                  new ArrayList<>(),
                  new ArrayList<>(),
                  new ArrayList<>(),
                  new ArrayList<>(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      recognizerOptions.runningMode(),
                      packets.get(HAND_GESTURES_OUT_STREAM_INDEX)));
            }
            return GestureRecognizerResult.create(
                PacketGetter.getProtoVector(
                    packets.get(LANDMARKS_OUT_STREAM_INDEX), NormalizedLandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(WORLD_LANDMARKS_OUT_STREAM_INDEX), LandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(HANDEDNESS_OUT_STREAM_INDEX), ClassificationList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(HAND_GESTURES_OUT_STREAM_INDEX), ClassificationList.parser()),
                BaseVisionTaskApi.generateResultTimestampMs(
                    recognizerOptions.runningMode(), packets.get(HAND_GESTURES_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    recognizerOptions.resultListener().ifPresent(handler::setResultListener);
    recognizerOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<GestureRecognizerOptions>builder()
                .setTaskName(GestureRecognizer.class.getSimpleName())
                .setTaskRunningModeName(recognizerOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(recognizerOptions)
                .setEnableFlowLimiting(recognizerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new GestureRecognizer(runner, recognizerOptions.runningMode());
  }

  /**
   * Constructor to initialize an {@link GestureRecognizer} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private GestureRecognizer(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs gesture recognition on the provided single image with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * GestureRecognizer} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public GestureRecognizerResult recognize(MPImage image) {
    return recognize(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs gesture recognition on the provided single image. Only use this method when the {@link
   * GestureRecognizer} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
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
  public GestureRecognizerResult recognize(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (GestureRecognizerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs gesture recognition on the provided video frame with default image processing options,
   * i.e. without any rotation applied. Only use this method when the {@link GestureRecognizer} is
   * created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public GestureRecognizerResult recognizeForVideo(MPImage image, long timestampMs) {
    return recognizeForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs gesture recognition on the provided video frame. Only use this method when the {@link
   * GestureRecognizer} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
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
  public GestureRecognizerResult recognizeForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (GestureRecognizerResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform gesture recognition with default image processing options,
   * i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link GestureRecognizerOptions}. Only use this method when the
   * {@link GestureRecognition} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the gesture recognizer. The input timestamps must be monotonically increasing.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void recognizeAsync(MPImage image, long timestampMs) {
    recognizeAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform gesture recognition, and the results will be available via the
   * {@link ResultListener} provided in the {@link GestureRecognizerOptions}. Only use this method
   * when the {@link GestureRecognition} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the gesture recognizer. The input timestamps must be monotonically increasing.
   *
   * <p>{@link GestureRecognizer} supports the following color space types:
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
  public void recognizeAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  /** Options for setting up an {@link GestureRecognizer}. */
  @AutoValue
  public abstract static class GestureRecognizerOptions extends TaskOptions {

    /** Builder for {@link GestureRecognizerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the gesture recognizer task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the gesture recognizer task. Default to the image mode. Gesture
       * recognizer has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for recognizing gestures on single image inputs.
       *   <li>VIDEO: The mode for recognizing gestures on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for recognizing gestures on a live stream of input data,
       *       such as from camera. In this mode, {@code setResultListener} must be called to set up
       *       a listener to receive the recognition results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /** Sets the maximum number of hands can be detected by the GestureRecognizer. */
      public abstract Builder setNumHands(Integer value);

      /** Sets minimum confidence score for the hand detection to be considered successful */
      public abstract Builder setMinHandDetectionConfidence(Float value);

      /** Sets minimum confidence score of hand presence score in the hand landmark detection. */
      public abstract Builder setMinHandPresenceConfidence(Float value);

      /** Sets the minimum confidence score for the hand tracking to be considered successful. */
      public abstract Builder setMinTrackingConfidence(Float value);

      /**
       * Sets the optional {@link ClassifierOptions} controlling the canned gestures classifier,
       * such as score threshold, allow list and deny list of gestures. The categories
       * for canned gesture classifiers are: ["None", "Closed_Fist", "Open_Palm",
       * "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
       *
       * <p>TODO  Note this option is subject to change, after scoring merging
       * calculator is implemented.
       */
      public abstract Builder setCannedGesturesClassifierOptions(
          ClassifierOptions classifierOptions);

      /**
       * Sets the optional {@link ClassifierOptions} controlling the custom gestures classifier,
       * such as score threshold, allow list and deny list of gestures.
       *
       * <p>TODO  Note this option is subject to change, after scoring merging
       * calculator is implemented.
       */
      public abstract Builder setCustomGesturesClassifierOptions(
          ClassifierOptions classifierOptions);

      /**
       * Sets the result listener to receive the detection results asynchronously when the gesture
       * recognizer is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<GestureRecognizerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract GestureRecognizerOptions autoBuild();

      /**
       * Validates and builds the {@link GestureRecognizerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the object detector is
       *     in the live stream mode.
       */
      public final GestureRecognizerOptions build() {
        GestureRecognizerOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The gesture recognizer is in the live stream mode, a user-defined result listener"
                    + " must be provided in GestureRecognizerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The gesture recognizer is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in GestureRecognizerOptions.");
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

    abstract Optional<ClassifierOptions> cannedGesturesClassifierOptions();

    abstract Optional<ClassifierOptions> customGesturesClassifierOptions();

    abstract Optional<ResultListener<GestureRecognizerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_GestureRecognizer_GestureRecognizerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setNumHands(1)
          .setMinHandDetectionConfidence(0.5f)
          .setMinHandPresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f);
    }

    /**
     * Converts a {@link GestureRecognizerOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      GestureRecognizerGraphOptionsProto.GestureRecognizerGraphOptions.Builder taskOptionsBuilder =
          GestureRecognizerGraphOptionsProto.GestureRecognizerGraphOptions.newBuilder()
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
      HandLandmarkerGraphOptionsProto.HandLandmarkerGraphOptions.Builder
          handLandmarkerGraphOptionsBuilder =
              HandLandmarkerGraphOptionsProto.HandLandmarkerGraphOptions.newBuilder();
      minTrackingConfidence()
          .ifPresent(handLandmarkerGraphOptionsBuilder::setMinTrackingConfidence);
      handLandmarkerGraphOptionsBuilder
          .setHandDetectorGraphOptions(handDetectorGraphOptionsBuilder.build())
          .setHandLandmarksDetectorGraphOptions(handLandmarksDetectorGraphOptionsBuilder.build());

      // Setup HandGestureRecognizerGraphOptions.
      HandGestureRecognizerGraphOptionsProto.HandGestureRecognizerGraphOptions.Builder
          handGestureRecognizerGraphOptionsBuilder =
              HandGestureRecognizerGraphOptionsProto.HandGestureRecognizerGraphOptions.newBuilder();
      cannedGesturesClassifierOptions()
          .ifPresent(
              classifierOptions -> {
                handGestureRecognizerGraphOptionsBuilder.setCannedGestureClassifierGraphOptions(
                    GestureClassifierGraphOptionsProto.GestureClassifierGraphOptions.newBuilder()
                        .setClassifierOptions(classifierOptions.convertToProto())
                        .build());
              });
      customGesturesClassifierOptions()
          .ifPresent(
              classifierOptions -> {
                handGestureRecognizerGraphOptionsBuilder.setCustomGestureClassifierGraphOptions(
                    GestureClassifierGraphOptionsProto.GestureClassifierGraphOptions.newBuilder()
                        .setClassifierOptions(classifierOptions.convertToProto())
                        .build());
              });
      taskOptionsBuilder
          .setHandLandmarkerGraphOptions(handLandmarkerGraphOptionsBuilder.build())
          .setHandGestureRecognizerGraphOptions(handGestureRecognizerGraphOptionsBuilder.build());
      return CalculatorOptions.newBuilder()
          .setExtension(
              GestureRecognizerGraphOptionsProto.GestureRecognizerGraphOptions.ext,
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
      throw new IllegalArgumentException("GestureRecognizer doesn't support region-of-interest.");
    }
  }
}
