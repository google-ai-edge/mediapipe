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

package com.google.mediapipe.tasks.vision.poselandmarker;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
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
import com.google.mediapipe.tasks.vision.posedetector.proto.PoseDetectorGraphOptionsProto;
import com.google.mediapipe.tasks.vision.poselandmarker.proto.PoseLandmarkerGraphOptionsProto;
import com.google.mediapipe.tasks.vision.poselandmarker.proto.PoseLandmarksDetectorGraphOptionsProto;
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
 * Performs pose landmarks detection on images.
 *
 * <p>This API expects a pre-trained pose landmarks model asset bundle. See <TODO link
 * to the DevSite documentation page>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that pose landmarks detection runs on.
 *       </ul>
 *   <li>Output PoseLandmarkerResult {@link PoseLandmarkerResult}
 *       <ul>
 *         <li>A PoseLandmarkerResult containing pose landmarks.
 *       </ul>
 * </ul>
 */
public final class PoseLandmarker extends BaseVisionTaskApi {
  private static final String TAG = PoseLandmarker.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  private static final int LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int WORLD_LANDMARKS_OUT_STREAM_INDEX = 1;
  private static final int IMAGE_OUT_STREAM_INDEX = 2;
  private static int segmentationMasksOutStreamIndex = -1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.pose_landmarker.PoseLandmarkerGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model file and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the pose landmarks model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model file and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the pose landmarks model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a model buffer and the default {@link
   * PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, PoseLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link PoseLandmarker} instance from a {@link PoseLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param landmarkerOptions a {@link PoseLandmarkerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link PoseLandmarker} creation.
   */
  public static PoseLandmarker createFromOptions(
      Context context, PoseLandmarkerOptions landmarkerOptions) {
    List<String> outputStreams = new ArrayList<>();
    outputStreams.add("NORM_LANDMARKS:pose_landmarks");
    outputStreams.add("WORLD_LANDMARKS:world_landmarks");
    outputStreams.add("IMAGE:image_out");
    if (landmarkerOptions.outputSegmentationMasks()) {
      outputStreams.add("SEGMENTATION_MASK:segmentation_masks");
      segmentationMasksOutStreamIndex = outputStreams.size() - 1;
    }

    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<PoseLandmarkerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<PoseLandmarkerResult, MPImage>() {
          @Override
          public PoseLandmarkerResult convertToTaskResult(List<Packet> packets) {
            // If there is no poses detected in the image, just returns empty lists.
            if (packets.get(LANDMARKS_OUT_STREAM_INDEX).isEmpty()) {
              return PoseLandmarkerResult.create(
                  new ArrayList<>(),
                  new ArrayList<>(),
                  Optional.empty(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
            }
            /** Get segmentation masks */
            Optional<List<MPImage>> segmentedMasks = Optional.empty();
            if (landmarkerOptions.outputSegmentationMasks()) {
              segmentedMasks = getSegmentationMasks(packets);
            }

            return PoseLandmarkerResult.create(
                PacketGetter.getProtoVector(
                    packets.get(LANDMARKS_OUT_STREAM_INDEX), NormalizedLandmarkList.parser()),
                PacketGetter.getProtoVector(
                    packets.get(WORLD_LANDMARKS_OUT_STREAM_INDEX), LandmarkList.parser()),
                segmentedMasks,
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
            TaskInfo.<PoseLandmarkerOptions>builder()
                .setTaskName(PoseLandmarker.class.getSimpleName())
                .setTaskRunningModeName(landmarkerOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(outputStreams)
                .setTaskOptions(landmarkerOptions)
                .setEnableFlowLimiting(landmarkerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new PoseLandmarker(runner, landmarkerOptions.runningMode());
  }

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> POSE_LANDMARKS = PoseLandmarksConnections.POSE_LANDMARKS;

  /**
   * Constructor to initialize a {@link PoseLandmarker} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private PoseLandmarker(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs pose landmarks detection on the provided single image with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * PoseLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs pose landmarks detection on the provided single image. Only use this method when the
   * {@link PoseLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java
   * doc for input image format.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
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
  public PoseLandmarkerResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (PoseLandmarkerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs pose landmarks detection on the provided video frame with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * PoseLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public PoseLandmarkerResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs pose landmarks detection on the provided video frame. Only use this method when the
   * {@link PoseLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
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
  public PoseLandmarkerResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (PoseLandmarkerResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform pose landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link PoseLandmarkerOptions}. Only use this method when the
   * {@link PoseLandmarker } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the pose landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
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
   * Sends live image data to perform pose landmarks detection, and the results will be available
   * via the {@link ResultListener} provided in the {@link PoseLandmarkerOptions}. Only use this
   * method when the {@link PoseLandmarker} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the pose landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link PoseLandmarker} supports the following color space types:
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

  /** Options for setting up an {@link PoseLandmarker}. */
  @AutoValue
  public abstract static class PoseLandmarkerOptions extends TaskOptions {

    /** Builder for {@link PoseLandmarkerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the pose landmarker task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the pose landmarker task. Default to the image mode. Pose
       * landmarker has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting pose landmarks on single image inputs.
       *   <li>VIDEO: The mode for detecting pose landmarks on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting pose landmarks on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /** Sets the maximum number of poses can be detected by the PoseLandmarker. */
      public abstract Builder setNumPoses(Integer value);

      /** Sets minimum confidence score for the pose detection to be considered successful */
      public abstract Builder setMinPoseDetectionConfidence(Float value);

      /** Sets minimum confidence score of pose presence score in the pose landmark detection. */
      public abstract Builder setMinPosePresenceConfidence(Float value);

      /** Sets the minimum confidence score for the pose tracking to be considered successful. */
      public abstract Builder setMinTrackingConfidence(Float value);

      public abstract Builder setOutputSegmentationMasks(boolean value);

      /**
       * Sets the result listener to receive the detection results asynchronously when the pose
       * landmarker is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<PoseLandmarkerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract PoseLandmarkerOptions autoBuild();

      /**
       * Validates and builds the {@link PoseLandmarkerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the pose landmarker is
       *     in the live stream mode.
       */
      public final PoseLandmarkerOptions build() {
        PoseLandmarkerOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The pose landmarker is in the live stream mode, a user-defined result listener"
                    + " must be provided in PoseLandmarkerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The pose landmarker is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in PoseLandmarkerOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<Integer> numPoses();

    abstract Optional<Float> minPoseDetectionConfidence();

    abstract Optional<Float> minPosePresenceConfidence();

    abstract Optional<Float> minTrackingConfidence();

    abstract boolean outputSegmentationMasks();

    abstract Optional<ResultListener<PoseLandmarkerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_PoseLandmarker_PoseLandmarkerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setNumPoses(1)
          .setMinPoseDetectionConfidence(0.5f)
          .setMinPosePresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setOutputSegmentationMasks(false);
    }

    /** Converts a {@link PoseLandmarkerOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.Builder taskOptionsBuilder =
          PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      // Setup PoseDetectorGraphOptions.
      PoseDetectorGraphOptionsProto.PoseDetectorGraphOptions.Builder
          poseDetectorGraphOptionsBuilder =
              PoseDetectorGraphOptionsProto.PoseDetectorGraphOptions.newBuilder();
      numPoses().ifPresent(poseDetectorGraphOptionsBuilder::setNumPoses);
      minPoseDetectionConfidence()
          .ifPresent(poseDetectorGraphOptionsBuilder::setMinDetectionConfidence);

      // Setup PoseLandmarkerGraphOptions.
      PoseLandmarksDetectorGraphOptionsProto.PoseLandmarksDetectorGraphOptions.Builder
          poseLandmarksDetectorGraphOptionsBuilder =
              PoseLandmarksDetectorGraphOptionsProto.PoseLandmarksDetectorGraphOptions.newBuilder();
      minPosePresenceConfidence()
          .ifPresent(poseLandmarksDetectorGraphOptionsBuilder::setMinDetectionConfidence);
      minTrackingConfidence().ifPresent(taskOptionsBuilder::setMinTrackingConfidence);

      taskOptionsBuilder
          .setPoseDetectorGraphOptions(poseDetectorGraphOptionsBuilder.build())
          .setPoseLandmarksDetectorGraphOptions(poseLandmarksDetectorGraphOptionsBuilder.build());

      return CalculatorOptions.newBuilder()
          .setExtension(
              PoseLandmarkerGraphOptionsProto.PoseLandmarkerGraphOptions.ext,
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
      throw new IllegalArgumentException("PoseLandmarker doesn't support region-of-interest.");
    }
  }

  private static Optional<List<MPImage>> getSegmentationMasks(List<Packet> packets) {
    Optional<List<MPImage>> segmentedMasks = Optional.of(new ArrayList<>());
    int width =
        PacketGetter.getImageWidthFromImageList(packets.get(segmentationMasksOutStreamIndex));
    int height =
        PacketGetter.getImageHeightFromImageList(packets.get(segmentationMasksOutStreamIndex));
    int imageListSize = PacketGetter.getImageListSize(packets.get(segmentationMasksOutStreamIndex));
    ByteBuffer[] buffersArray = new ByteBuffer[imageListSize];

    // Segmentation mask is a float type image.
    int numBytes = 4;
    for (int i = 0; i < imageListSize; i++) {
      buffersArray[i] = ByteBuffer.allocateDirect(width * height * numBytes);
    }

    if (!PacketGetter.getImageList(
        packets.get(segmentationMasksOutStreamIndex),
        buffersArray,
        /** deepCopy= */
        true)) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL.ordinal(),
          "There is an error getting segmented masks.");
    }
    for (ByteBuffer buffer : buffersArray) {
      ByteBufferImageBuilder builder =
          new ByteBufferImageBuilder(buffer, width, height, MPImage.IMAGE_FORMAT_VEC32F1);
      segmentedMasks.get().add(builder.build());
    }
    return segmentedMasks;
  }
}
