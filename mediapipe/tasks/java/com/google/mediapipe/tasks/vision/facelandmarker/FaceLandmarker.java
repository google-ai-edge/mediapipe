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

package com.google.mediapipe.tasks.vision.facelandmarker;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import com.google.auto.value.AutoValue;
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
import com.google.mediapipe.tasks.vision.facedetector.proto.FaceDetectorGraphOptionsProto;
import com.google.mediapipe.tasks.vision.facegeometry.proto.FaceGeometryProto.FaceGeometry;
import com.google.mediapipe.tasks.vision.facelandmarker.proto.FaceLandmarkerGraphOptionsProto;
import com.google.mediapipe.tasks.vision.facelandmarker.proto.FaceLandmarksDetectorGraphOptionsProto;
import com.google.mediapipe.formats.proto.MatrixDataProto.MatrixData;
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
 * Performs face landmarks detection on images.
 *
 * <p>This API expects a pre-trained face landmarks model asset bundle. See <TODO link
 * to the DevSite documentation page>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that face landmarks detection runs on.
 *       </ul>
 *   <li>Output {@link FaceLandmarkerResult}
 *       <ul>
 *         <li>A FaceLandmarkerResult containing face landmarks.
 *       </ul>
 * </ul>
 */
public final class FaceLandmarker extends BaseVisionTaskApi {
  private static final String TAG = FaceLandmarker.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  private static final int LANDMARKS_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;
  private static int blendshapesOutStreamIndex = -1;
  private static int faceGeometryOutStreamIndex = -1;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.face_landmarker.FaceLandmarkerGraph";

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates a {@link FaceLandmarker} instance from a model asset bundle path and the default {@link
   * FaceLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelAssetPath path to the face landmarks model with metadata in the assets.
   * @throws MediaPipeException if there is an error during {@link FaceLandmarker} creation.
   */
  public static FaceLandmarker createFromFile(Context context, String modelAssetPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelAssetPath).build();
    return createFromOptions(
        context, FaceLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link FaceLandmarker} instance from a model asset bundle file and the default {@link
   * FaceLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelAssetFile the face landmarks model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link FaceLandmarker} creation.
   */
  public static FaceLandmarker createFromFile(Context context, File modelAssetFile)
      throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelAssetFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, FaceLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates a {@link FaceLandmarker} instance from a model asset bundle buffer and the default
   * {@link FaceLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model.
   * @throws MediaPipeException if there is an error during {@link FaceLandmarker} creation.
   */
  public static FaceLandmarker createFromBuffer(
      Context context, final ByteBuffer modelAssetBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelAssetBuffer).build();
    return createFromOptions(
        context, FaceLandmarkerOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates a {@link FaceLandmarker} instance from a {@link FaceLandmarkerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param landmarkerOptions a {@link FaceLandmarkerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link FaceLandmarker} creation.
   */
  public static FaceLandmarker createFromOptions(
      Context context, FaceLandmarkerOptions landmarkerOptions) {
    List<String> outputStreams = new ArrayList<>();
    outputStreams.add("NORM_LANDMARKS:face_landmarks");
    outputStreams.add("IMAGE:image_out");
    if (landmarkerOptions.outputFaceBlendshapes()) {
      outputStreams.add("BLENDSHAPES:face_blendshapes");
      blendshapesOutStreamIndex = outputStreams.size() - 1;
    }
    if (landmarkerOptions.outputFacialTransformationMatrixes()) {
      outputStreams.add("FACE_GEOMETRY:face_geometry");
      faceGeometryOutStreamIndex = outputStreams.size() - 1;
    }
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<FaceLandmarkerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<FaceLandmarkerResult, MPImage>() {
          @Override
          public FaceLandmarkerResult convertToTaskResult(List<Packet> packets) {
            // If there is no faces detected in the image, just returns empty lists.
            if (packets.get(LANDMARKS_OUT_STREAM_INDEX).isEmpty()) {
              return FaceLandmarkerResult.create(
                  new ArrayList<>(),
                  Optional.empty(),
                  Optional.empty(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      landmarkerOptions.runningMode(), packets.get(LANDMARKS_OUT_STREAM_INDEX)));
            }

            Optional<List<ClassificationList>> blendshapes = Optional.empty();
            if (landmarkerOptions.outputFaceBlendshapes()) {
              blendshapes =
                  Optional.of(
                      PacketGetter.getProtoVector(
                          packets.get(blendshapesOutStreamIndex), ClassificationList.parser()));
            }

            Optional<List<MatrixData>> facialTransformationMatrixes = Optional.empty();
            if (landmarkerOptions.outputFacialTransformationMatrixes()) {
              List<FaceGeometry> faceGeometryList =
                  PacketGetter.getProtoVector(
                      packets.get(faceGeometryOutStreamIndex), FaceGeometry.parser());
              facialTransformationMatrixes = Optional.of(new ArrayList<>());
              for (FaceGeometry faceGeometry : faceGeometryList) {
                facialTransformationMatrixes.get().add(faceGeometry.getPoseTransformMatrix());
              }
            }

            return FaceLandmarkerResult.create(
                PacketGetter.getProtoVector(
                    packets.get(LANDMARKS_OUT_STREAM_INDEX), NormalizedLandmarkList.parser()),
                blendshapes,
                facialTransformationMatrixes,
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
            TaskInfo.<FaceLandmarkerOptions>builder()
                .setTaskName(FaceLandmarker.class.getSimpleName())
                .setTaskRunningModeName(landmarkerOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(outputStreams)
                .setTaskOptions(landmarkerOptions)
                .setEnableFlowLimiting(landmarkerOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new FaceLandmarker(runner, landmarkerOptions.runningMode());
  }

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_LIPS =
      FaceLandmarksConnections.FACE_LANDMARKS_LIPS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_LEFT_EYE =
      FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_LEFT_EYE_BROW =
      FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE_BROW;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_LEFT_IRIS =
      FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_RIGHT_EYE =
      FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_RIGHT_EYE_BROW =
      FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE_BROW;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_RIGHT_IRIS =
      FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_FACE_OVAL =
      FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_CONNECTORS =
      FaceLandmarksConnections.FACE_LANDMARKS_CONNECTORS;

  @SuppressWarnings("ConstantCaseForConstants")
  public static final Set<Connection> FACE_LANDMARKS_TESSELATION =
      FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION;

  /**
   * Constructor to initialize an {@link FaceLandmarker} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private FaceLandmarker(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs face landmarks detection on the provided single image with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * FaceLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public FaceLandmarkerResult detect(MPImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs face landmarks detection on the provided single image. Only use this method when the
   * {@link FaceLandmarker} is created with {@link RunningMode.IMAGE}. TODO update java
   * doc for input image format.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
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
  public FaceLandmarkerResult detect(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (FaceLandmarkerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs face landmarks detection on the provided video frame with default image processing
   * options, i.e. without any rotation applied. Only use this method when the {@link
   * FaceLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public FaceLandmarkerResult detectForVideo(MPImage image, long timestampMs) {
    return detectForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs face landmarks detection on the provided video frame. Only use this method when the
   * {@link FaceLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
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
  public FaceLandmarkerResult detectForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    return (FaceLandmarkerResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform face landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link FaceLandmarkerOptions}. Only use this method when the
   * {@link FaceLandmarker } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the face landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
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
   * Sends live image data to perform face landmarks detection, and the results will be available
   * via the {@link ResultListener} provided in the {@link FaceLandmarkerOptions}. Only use this
   * method when the {@link FaceLandmarker} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the face landmarker. The input timestamps must be monotonically increasing.
   *
   * <p>{@link FaceLandmarker} supports the following color space types:
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

  /** Options for setting up an {@link FaceLandmarker}. */
  @AutoValue
  public abstract static class FaceLandmarkerOptions extends TaskOptions {

    /** Builder for {@link FaceLandmarkerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the face landmarker task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the face landmarker task. Default to the image mode. Hand
       * landmarker has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for detecting face landmarks on single image inputs.
       *   <li>VIDEO: The mode for detecting face landmarks on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for detecting face landmarks on a live stream of input
       *       data, such as from camera. In this mode, {@code setResultListener} must be called to
       *       set up a listener to receive the detection results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /** Sets the maximum number of faces can be detected by the FaceLandmarker. */
      public abstract Builder setNumFaces(Integer value);

      /** Sets minimum confidence score for the face detection to be considered successful */
      public abstract Builder setMinFaceDetectionConfidence(Float value);

      /** Sets minimum confidence score of face presence score in the face landmark detection. */
      public abstract Builder setMinFacePresenceConfidence(Float value);

      /** Sets the minimum confidence score for the face tracking to be considered successful. */
      public abstract Builder setMinTrackingConfidence(Float value);

      /**
       * Whether FaceLandmarker outputs face blendshapes classification. Face blendshapes are used
       * for rendering the 3D face model.
       */
      public abstract Builder setOutputFaceBlendshapes(Boolean value);

      /**
       * Whether FaceLandmarker outptus facial transformation_matrix. Facial transformation matrix
       * is used to transform the face landmarks in canonical face to the detected face, so that
       * users can apply face effects on the detected landmarks.
       */
      public abstract Builder setOutputFacialTransformationMatrixes(Boolean value);

      /**
       * Sets the result listener to receive the detection results asynchronously when the face
       * landmarker is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<FaceLandmarkerResult, MPImage> value);

      /** Sets an optional error listener. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract FaceLandmarkerOptions autoBuild();

      /**
       * Validates and builds the {@link FaceLandmarkerOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the face landmarker is
       *     in the live stream mode.
       */
      public final FaceLandmarkerOptions build() {
        FaceLandmarkerOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The face landmarker is in the live stream mode, a user-defined result listener"
                    + " must be provided in FaceLandmarkerOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The face landmarker is in the image or the video mode, a user-defined result"
                  + " listener shouldn't be provided in FaceLandmarkerOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract Optional<Integer> numFaces();

    abstract Optional<Float> minFaceDetectionConfidence();

    abstract Optional<Float> minFacePresenceConfidence();

    abstract Optional<Float> minTrackingConfidence();

    abstract Boolean outputFaceBlendshapes();

    abstract Boolean outputFacialTransformationMatrixes();

    abstract Optional<ResultListener<FaceLandmarkerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_FaceLandmarker_FaceLandmarkerOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setNumFaces(1)
          .setMinFaceDetectionConfidence(0.5f)
          .setMinFacePresenceConfidence(0.5f)
          .setMinTrackingConfidence(0.5f)
          .setOutputFaceBlendshapes(false)
          .setOutputFacialTransformationMatrixes(false);
    }

    /** Converts a {@link FaceLandmarkerOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      FaceLandmarkerGraphOptionsProto.FaceLandmarkerGraphOptions.Builder taskOptionsBuilder =
          FaceLandmarkerGraphOptionsProto.FaceLandmarkerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      // Setup FaceDetectorGraphOptions.
      FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions.Builder
          faceDetectorGraphOptionsBuilder =
              FaceDetectorGraphOptionsProto.FaceDetectorGraphOptions.newBuilder();
      numFaces().ifPresent(faceDetectorGraphOptionsBuilder::setNumFaces);
      minFaceDetectionConfidence()
          .ifPresent(faceDetectorGraphOptionsBuilder::setMinDetectionConfidence);

      // Setup FaceLandmarkerGraphOptions.
      FaceLandmarksDetectorGraphOptionsProto.FaceLandmarksDetectorGraphOptions.Builder
          faceLandmarksDetectorGraphOptionsBuilder =
              FaceLandmarksDetectorGraphOptionsProto.FaceLandmarksDetectorGraphOptions.newBuilder();
      minFacePresenceConfidence()
          .ifPresent(faceLandmarksDetectorGraphOptionsBuilder::setMinDetectionConfidence);
      minTrackingConfidence().ifPresent(taskOptionsBuilder::setMinTrackingConfidence);

      taskOptionsBuilder
          .setFaceDetectorGraphOptions(faceDetectorGraphOptionsBuilder.build())
          .setFaceLandmarksDetectorGraphOptions(faceLandmarksDetectorGraphOptionsBuilder.build());

      return CalculatorOptions.newBuilder()
          .setExtension(
              FaceLandmarkerGraphOptionsProto.FaceLandmarkerGraphOptions.ext,
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
      throw new IllegalArgumentException("FaceLandmarker doesn't support region-of-interest.");
    }
  }
}
