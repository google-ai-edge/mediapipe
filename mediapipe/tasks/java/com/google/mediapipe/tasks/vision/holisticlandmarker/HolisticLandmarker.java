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

package com.google.mediapipe.tasks.vision.holisticlandmarker;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.formats.proto.LandmarkProto.LandmarkList;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.formats.proto.ClassificationProto.ClassificationList;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
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
import com.google.mediapipe.tasks.vision.facedetector.proto.FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions;
import com.google.mediapipe.tasks.vision.facelandmarker.proto.FaceLandmarksDetectorGraphOptionsProto.FaceLandmarksDetectorGraphOptions;
import com.google.mediapipe.tasks.vision.handlandmarker.proto.HandLandmarksDetectorGraphOptionsProto.HandLandmarksDetectorGraphOptions;
import com.google.mediapipe.tasks.vision.holisticlandmarker.proto.HolisticLandmarkerGraphOptionsProto.HolisticLandmarkerGraphOptions;
import com.google.mediapipe.tasks.vision.posedetector.proto.PoseDetectorGraphOptionsProto.PoseDetectorGraphOptions;
import com.google.mediapipe.tasks.vision.poselandmarker.proto.PoseLandmarksDetectorGraphOptionsProto.PoseLandmarksDetectorGraphOptions;
import com.google.protobuf.Any;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs holistic landmarks detection on images.
 *
 * <p>This API expects a pre-trained holistic landmarks model asset bundle.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that holistic landmarks detection runs on.
 *       </ul>
 *   <li>Output {@link HolisticLandmarkerResult}
 *       <ul>
 *         <li>A HolisticLandmarkerResult containing holistic landmarks.
 *       </ul>
 * </ul>
 */
public final class HolisticLandmarker extends BaseVisionTaskApi {
  private static final String TAG = HolisticLandmarker.class.getSimpleName();

  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String POSE_LANDMARKS_STREAM = "pose_landmarks";
  private static final String POSE_WORLD_LANDMARKS_STREAM = "pose_world_landmarks";
  private static final String POSE_SEGMENTATION_MASK_STREAM = "pose_segmentation_mask";
  private static final String FACE_LANDMARKS_STREAM = "face_landmarks";
  private static final String FACE_BLENDSHAPES_STREAM = "extra_blendshapes";
  private static final String LEFT_HAND_LANDMARKS_STREAM = "left_hand_landmarks";
  private static final String LEFT_HAND_WORLD_LANDMARKS_STREAM = "left_hand_world_landmarks";
  private static final String RIGHT_HAND_LANDMARKS_STREAM = "right_hand_landmarks";
  private static final String RIGHT_HAND_WORLD_LANDMARKS_STREAM = "right_hand_world_landmarks";
  private static final String IMAGE_OUT_STREAM_NAME = "image_out";

  private static final int FACE_LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int POSE_LANDMARKS_OUT_STREAM_INDEX = 1;
  private static final int POSE_WORLD_LANDMARKS_OUT_STREAM_INDEX = 2;
  private static final int LEFT_HAND_LANDMARKS_OUT_STREAM_INDEX = 3;
  private static final int LEFT_HAND_WORLD_LANDMARKS_OUT_STREAM_INDEX = 4;
  private static final int RIGHT_HAND_LANDMARKS_OUT_STREAM_INDEX = 5;
  private static final int RIGHT_HAND_WORLD_LANDMARKS_OUT_STREAM_INDEX = 6;
  private static final int IMAGE_OUT_STREAM_INDEX = 7;

  private static final float DEFAULT_PRESENCE_THRESHOLD = 0.5f;
  private static final float DEFAULT_SUPPRESION_THRESHOLD = 0.3f;
  private static final boolean DEFAULT_OUTPUT_FACE_BLENDSHAPES = false;
  private static final boolean DEFAULT_OUTPUT_SEGMENTATION_MASKS = false;

  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.holistic_landmarker.HolisticLandmarkerGraph";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME));

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link HolisticLandmarker} instance from a model asset bundle path and the default
   * {@link HolisticLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelAssetPath path to the holistic landmarks model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link HolisticLandmarker} creation.
   */
  public static HolisticLandmarker createFromFile(Context context, String modelAssetPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelAssetPath).build();
    return createFromOptions(
        context, HolisticLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link HolisticLandmarker} instance from a model asset bundle file and the default
   * {@link HolisticLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelAssetFile the holistic landmarks model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link HolisticLandmarker} creation.
   */
  public static HolisticLandmarker createFromFile(Context context, File modelAssetFile)
      throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelAssetFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, HolisticLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link HolisticLandmarker} instance from a model asset bundle buffer and the default
   * {@link HolisticLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelAssetBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the
   *     detection model.
   * @throws MediaPipeException if there is an error during {@link HolisticLandmarker} creation.
   */
  public static HolisticLandmarker createFromBuffer(
      Context context, final ByteBuffer modelAssetBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelAssetBuffer).build();
    return createFromOptions(
        context, HolisticLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link HolisticLandmarker} instance from a {@link HolisticLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param landmarkerOptions a {@link HolisticLandmarkerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link HolisticLandmarker} creation.
   */
  public static HolisticLandmarker createFromOptions(
      Context context, HolisticLandmarkerOptions landmarkerOptions) {
    List<String> outputStreams = new ArrayList<>();
    outputStreams.add("FACE_LANDMARKS:" + FACE_LANDMARKS_STREAM);
    outputStreams.add("POSE_LANDMARKS:" + POSE_LANDMARKS_STREAM);
    outputStreams.add("POSE_WORLD_LANDMARKS:" + POSE_WORLD_LANDMARKS_STREAM);
    outputStreams.add("LEFT_HAND_LANDMARKS:" + LEFT_HAND_LANDMARKS_STREAM);
    outputStreams.add("LEFT_HAND_WORLD_LANDMARKS:" + LEFT_HAND_WORLD_LANDMARKS_STREAM);
    outputStreams.add("RIGHT_HAND_LANDMARKS:" + RIGHT_HAND_LANDMARKS_STREAM);
    outputStreams.add("RIGHT_HAND_WORLD_LANDMARKS:" + RIGHT_HAND_WORLD_LANDMARKS_STREAM);
    outputStreams.add("IMAGE:" + IMAGE_OUT_STREAM_NAME);

    int[] faceBlendshapesOutStreamIndex = new int[] {-1};
    if (landmarkerOptions.outputFaceBlendshapes()) {
      outputStreams.add("FACE_BLENDSHAPES:" + FACE_BLENDSHAPES_STREAM);
      faceBlendshapesOutStreamIndex[0] = outputStreams.size() - 1;
    }

    int[] poseSegmentationMasksOutStreamIndex = new int[] {-1};
    if (landmarkerOptions.outputPoseSegmentationMasks()) {
      outputStreams.add("POSE_SEGMENTATION_MASK:" + POSE_SEGMENTATION_MASK_STREAM);
      poseSegmentationMasksOutStreamIndex[0] = outputStreams.size() - 1;
    }

    OutputHandler<HolisticLandmarkerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<HolisticLandmarkerResult, MPImage>() {
          @Override
          public HolisticLandmarkerResult convertToTaskResult(List<Packet> packets) {
            NormalizedLandmarkList faceLandmarkProtos =
                getNormalizedLandmarkList(packets.get(FACE_LANDMARKS_OUT_STREAM_INDEX));
            Optional<ClassificationList> faceBlendshapeProtos =
                landmarkerOptions.outputFaceBlendshapes()
                    ? Optional.of(
                        PacketGetter.getProto(
                            packets.get(faceBlendshapesOutStreamIndex[0]),
                            ClassificationList.parser()))
                    : Optional.empty();
            NormalizedLandmarkList poseLandmarkProtos =
                getNormalizedLandmarkList(packets.get(POSE_LANDMARKS_OUT_STREAM_INDEX));
            LandmarkList poseWorldLandmarkProtos =
                getLandmarkList(packets.get(POSE_WORLD_LANDMARKS_OUT_STREAM_INDEX));
            Optional<MPImage> segmentationMask =
                landmarkerOptions.outputPoseSegmentationMasks()
                    ? Optional.of(
                        getSegmentationMask(packets, poseSegmentationMasksOutStreamIndex[0]))
                    : Optional.empty();
            NormalizedLandmarkList leftHandLandmarkProtos =
                getNormalizedLandmarkList(packets.get(LEFT_HAND_LANDMARKS_OUT_STREAM_INDEX));
            LandmarkList leftHandWorldLandmarkProtos =
                getLandmarkList(packets.get(LEFT_HAND_WORLD_LANDMARKS_OUT_STREAM_INDEX));
            NormalizedLandmarkList rightHandLandmarkProtos =
                getNormalizedLandmarkList(packets.get(RIGHT_HAND_LANDMARKS_OUT_STREAM_INDEX));
            LandmarkList rightHandWorldLandmarkProtos =
                getLandmarkList(packets.get(RIGHT_HAND_WORLD_LANDMARKS_OUT_STREAM_INDEX));

            return HolisticLandmarkerResult.create(
                faceLandmarkProtos,
                faceBlendshapeProtos,
                poseLandmarkProtos,
                poseWorldLandmarkProtos,
                segmentationMask,
                leftHandLandmarkProtos,
                leftHandWorldLandmarkProtos,
                rightHandLandmarkProtos,
                rightHandWorldLandmarkProtos,
                BaseVisionTaskApi.generateResultTimestampMs(
                    landmarkerOptions.runningMode(), packets.get(FACE_LANDMARKS_OUT_STREAM_INDEX)));
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
            TaskInfo.<HolisticLandmarkerOptions>builder()
                .setTaskName(HolisticLandmarker.class.getSimpleName())
                .setTaskRunningModeName(landmarkerOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(outputStreams)
                .setTaskOptions(landmarkerOptions)
                .setEnableFlowLimiting(landmarkerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new HolisticLandmarker(runner, landmarkerOptions.runningMode());
  }

  /**
   * Constructor to initialize an {@link HolisticLandmarker} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private HolisticLandmarker(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, /* normRectStreamName= */ "");
  }

  /**
   * Performs holistic landmarks detection on the provided single image with default image
   * processing options, i.e. without any rotation applied. Only use this method when the {@link
   * HolisticLandmarker} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public HolisticLandmarkerResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs holistic landmarks detection on the provided single image. Only use this method when
   * the {@link HolisticLandmarker} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
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
  public HolisticLandmarkerResult detect(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (HolisticLandmarkerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs holistic landmarks detection on the provided video frame with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * HolisticLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame"s timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public HolisticLandmarkerResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs holistic landmarks detection on the provided video frame. Only use this method when
   * the {@link HolisticLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame"s timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
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
  public HolisticLandmarkerResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (HolisticLandmarkerResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform holistic landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link HolisticLandmarkerOptions}. Only use this method when
   * the {@link HolisticLandmarker } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the holistic landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
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
   * Sends live image data to perform holistic landmarks detection, and the results will be
   * available via the {@link ResultListener} provided in the {@link HolisticLandmarkerOptions}.
   * Only use this method when the {@link HolisticLandmarker} is created with {@link
   * RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the holistic landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link HolisticLandmarker} supports the following color space types:
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

  /** Options for setting up an {@link HolisticLandmarker}. */
  @AutoValue
  public abstract static class HolisticLandmarkerOptions extends TaskOptions {

    /** Builder for {@link HolisticLandmarkerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the holistic landmarker task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the holistic landmarker task. Defaults to the image mode.
       * Holistic landmarker has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting holistic landmarks on single image inputs.
       *   <li>VIDEO: The mode for detecting holistic landmarks on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting holistic landmarks on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /**
       * Sets minimum confidence score for the face detection to be considered successful. Defaults
       * to 0.5.
       */
      public abstract Builder setMinFaceDetectionConfidence(Float value);

      /**
       * The minimum threshold for the face suppression score in the face detection. Defaults to
       * 0.3.
       */
      public abstract Builder setMinFaceSuppressionThreshold(Float value);

      /**
       * Sets minimum confidence score for the face landmark detection to be considered successful.
       * Defaults to 0.5.
       */
      public abstract Builder setMinFacePresenceConfidence(Float value);

      /**
       * The minimum confidence score for the pose detection to be considered successful. Defaults
       * to 0.5.
       */
      public abstract Builder setMinPoseDetectionConfidence(Float value);

      /**
       * The minimum threshold for the pose suppression score in the pose detection. Defaults to
       * 0.3.
       */
      public abstract Builder setMinPoseSuppressionThreshold(Float value);

      /**
       * The minimum confidence score for the pose landmarks detection to be considered successful.
       * Defaults to 0.5.
       */
      public abstract Builder setMinPosePresenceConfidence(Float value);

      /**
       * The minimum confidence score for the hand landmark detection to be considered successful.
       * Defaults to 0.5.
       */
      public abstract Builder setMinHandLandmarksConfidence(Float value);

      /** Whether to output segmentation masks. Defaults to false. */
      public abstract Builder setOutputPoseSegmentationMasks(Boolean value);

      /** Whether to output face blendshapes. Defaults to false. */
      public abstract Builder setOutputFaceBlendshapes(Boolean value);

      /**
       * Sets the result listener to receive the detection results asynchronously when the holistic
       * landmarker is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<HolisticLandmarkerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract HolisticLandmarkerOptions autoBuild();

      /**
       * Validates and builds the {@link HolisticLandmarkerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the holistic
       *     landmarker is in the live stream mode.
       */
      public final HolisticLandmarkerOptions build() {
        HolisticLandmarkerOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The holistic landmarker is in the live stream mode, a user-defined result listener"
                    + " must be provided in HolisticLandmarkerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The holistic landmarker is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in HolisticLandmarkerOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<Float> minFaceDetectionConfidence();

    abstract Optional<Float> minFaceSuppressionThreshold();

    abstract Optional<Float> minFacePresenceConfidence();

    abstract Optional<Float> minPoseDetectionConfidence();

    abstract Optional<Float> minPoseSuppressionThreshold();

    abstract Optional<Float> minPosePresenceConfidence();

    abstract Optional<Float> minHandLandmarksConfidence();

    abstract Boolean outputFaceBlendshapes();

    abstract Boolean outputPoseSegmentationMasks();

    abstract Optional<ResultListener<HolisticLandmarkerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_HolisticLandmarker_HolisticLandmarkerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setMinFaceDetectionConfidence(DEFAULT_PRESENCE_THRESHOLD)
          .setMinFaceSuppressionThreshold(DEFAULT_SUPPRESION_THRESHOLD)
          .setMinFacePresenceConfidence(DEFAULT_PRESENCE_THRESHOLD)
          .setMinPoseDetectionConfidence(DEFAULT_PRESENCE_THRESHOLD)
          .setMinPoseSuppressionThreshold(DEFAULT_SUPPRESION_THRESHOLD)
          .setMinPosePresenceConfidence(DEFAULT_PRESENCE_THRESHOLD)
          .setMinHandLandmarksConfidence(DEFAULT_PRESENCE_THRESHOLD)
          .setOutputFaceBlendshapes(DEFAULT_OUTPUT_FACE_BLENDSHAPES)
          .setOutputPoseSegmentationMasks(DEFAULT_OUTPUT_SEGMENTATION_MASKS);
    }

    /** Converts a {@link HolisticLandmarkerOptions} to a {@link Any} protobuf message. */
    @Override
    public Any convertToAnyProto() {
      HolisticLandmarkerGraphOptions.Builder holisticLandmarkerGraphOptions =
          HolisticLandmarkerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      HandLandmarksDetectorGraphOptions.Builder handLandmarksDetectorGraphOptions =
          HandLandmarksDetectorGraphOptions.newBuilder();
      FaceDetectorGraphOptions.Builder faceDetectorGraphOptions =
          FaceDetectorGraphOptions.newBuilder();
      FaceLandmarksDetectorGraphOptions.Builder faceLandmarksDetectorGraphOptions =
          FaceLandmarksDetectorGraphOptions.newBuilder();
      PoseDetectorGraphOptions.Builder poseDetectorGraphOptions =
          PoseDetectorGraphOptions.newBuilder();
      PoseLandmarksDetectorGraphOptions.Builder poseLandmarkerGraphOptions =
          PoseLandmarksDetectorGraphOptions.newBuilder();

      // Configure hand detector options.
      minHandLandmarksConfidence()
          .ifPresent(handLandmarksDetectorGraphOptions::setMinDetectionConfidence);

      // Configure pose detector options.
      minPoseDetectionConfidence().ifPresent(poseDetectorGraphOptions::setMinDetectionConfidence);
      minPoseSuppressionThreshold().ifPresent(poseDetectorGraphOptions::setMinSuppressionThreshold);
      minPosePresenceConfidence().ifPresent(poseLandmarkerGraphOptions::setMinDetectionConfidence);

      // Configure face detector options.
      minFaceDetectionConfidence().ifPresent(faceDetectorGraphOptions::setMinDetectionConfidence);
      minFaceSuppressionThreshold().ifPresent(faceDetectorGraphOptions::setMinSuppressionThreshold);
      minFacePresenceConfidence()
          .ifPresent(faceLandmarksDetectorGraphOptions::setMinDetectionConfidence);

      holisticLandmarkerGraphOptions
          .setHandLandmarksDetectorGraphOptions(handLandmarksDetectorGraphOptions.build())
          .setFaceDetectorGraphOptions(faceDetectorGraphOptions.build())
          .setFaceLandmarksDetectorGraphOptions(faceLandmarksDetectorGraphOptions.build())
          .setPoseDetectorGraphOptions(poseDetectorGraphOptions.build())
          .setPoseLandmarksDetectorGraphOptions(poseLandmarkerGraphOptions.build());

      return Any.newBuilder()
          .setTypeUrl(
              "type.googleapis.com/mediapipe.tasks.vision.holistic_landmarker.proto.HolisticLandmarkerGraphOptions")
          .setValue(holisticLandmarkerGraphOptions.build().toByteString())
          .build();
    }
  }

  /**
   * Validates that the provided {@link ImageProcessingOptions} doesn"t contain a
   * region-of-interest.
   */
  private static void validateImageProcessingOptions(
      ImageProcessingOptions imageProcessingOptions) {
    if (imageProcessingOptions.regionOfInterest().isPresent()) {
      throw new IllegalArgumentException("HolisticLandmarker doesn't support region-of-interest.");
    }
  }

  private static MPImage getSegmentationMask(List<Packet> packets, int packetIndex) {
    int width = PacketGetter.getImageWidth(packets.get(packetIndex));
    int height = PacketGetter.getImageHeight(packets.get(packetIndex));
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);

    if (!PacketGetter.getImageData(packets.get(packetIndex), buffer)) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.INTERNAL.ordinal(),
          "There was an error getting the sefmentation mask.");
    }

    ByteBufferImageBuilder builder =
        new ByteBufferImageBuilder(buffer, width, height, MPImage.IMAGE_FORMAT_VEC32F1);
    return builder.build();
  }

  private static NormalizedLandmarkList getNormalizedLandmarkList(Packet packet) {
    return packet.isEmpty()
        ? NormalizedLandmarkList.getDefaultInstance()
        : PacketGetter.getProto(packet, NormalizedLandmarkList.parser());
  }

  private static LandmarkList getLandmarkList(Packet packet) {
    return packet.isEmpty()
        ? LandmarkList.getDefaultInstance()
        : PacketGetter.getProto(packet, LandmarkList.parser());
  }
}
