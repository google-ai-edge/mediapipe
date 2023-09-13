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

package com.google.mediapipe.tasks.vision.facestylizer;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
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
import com.google.mediapipe.tasks.core.TaskResult;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facestylizer.proto.FaceStylizerGraphOptionsProto;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs face stylization on images.
 *
 * <p>Note that, in addition to the standard stylization API, {@link #stylize} and {@link
 * #stylizeForVideo}, that take an input image and return the outputs, but involves deep copy of the
 * returns, FaceStylizer also supports the callback API, {@link #stylizeWithResultListener} and
 * {@link #stylizeForVideoWithResultListener}, which allow you to access the outputs through zero
 * copy for the duration of the result listener.
 *
 * <p>The callback API is available for all {@link RunningMode} in FaceStylizer. Set {@link
 * ResultListener} in {@link FaceStylizerOptions} properly to use the callback API.
 *
 * <p>The API expects a TFLite model with,<a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that face stylizer runs on.
 *       </ul>
 *   <li>Output MPImage {@link MPImage}
 *       <ul>
 *         <li>A MPImage containing a stylized face.
 *       </ul>
 * </ul>
 */
public final class FaceStylizer extends BaseVisionTaskApi {
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final String IMAGE_OUT_STREAM_NAME = "image_out";

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));

  @SuppressWarnings("ConstantCaseForConstants")
  private static final List<String> OUTPUT_STREAMS =
      Collections.singletonList("STYLIZED_IMAGE:" + IMAGE_OUT_STREAM_NAME);

  private static final int IMAGE_OUT_STREAM_INDEX = 0;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.face_stylizer.FaceStylizerGraph";
  private final boolean hasResultListener;

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates an {@link FaceStylizer} instance from an {@link FaceStylizerOptions}.
   *
   * @param context an Android {@link Context}.
   * @param stylizerOptions an {@link FaceStylizerOptions} instance.
   * @throws MediaPipeException if there is an error during {@link FaceStylizer} creation.
   */
  public static FaceStylizer createFromOptions(
      Context context, FaceStylizerOptions stylizerOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<FaceStylizerResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<FaceStylizerResult, MPImage>() {
          @Override
          public FaceStylizerResult convertToTaskResult(List<Packet> packets)
              throws MediaPipeException {
            Packet packet = packets.get(IMAGE_OUT_STREAM_INDEX);
            if (packet.isEmpty()) {
              return FaceStylizerResult.create(
                  Optional.empty(),
                  BaseVisionTaskApi.generateResultTimestampMs(
                      RunningMode.IMAGE, packets.get(IMAGE_OUT_STREAM_INDEX)));
            }
            int width = PacketGetter.getImageWidth(packet);
            int height = PacketGetter.getImageHeight(packet);
            int numChannels = PacketGetter.getImageNumChannels(packet);
            int imageFormat =
                numChannels == 3 ? MPImage.IMAGE_FORMAT_RGB : MPImage.IMAGE_FORMAT_RGBA;

            ByteBuffer imageBuffer;
            // If resultListener is not provided, the resulted MPImage is deep copied from the
            // MediaPipe graph. If provided, the result MPImage is wrapping the MediaPipe packet
            // memory.
            if (!stylizerOptions.resultListener().isPresent()) {
              imageBuffer = ByteBuffer.allocateDirect(width * height * numChannels);
              if (!PacketGetter.getImageData(packet, imageBuffer)) {
                imageBuffer = null;
              }
            } else {
              imageBuffer = PacketGetter.getImageDataDirectly(packet);
            }

            if (imageBuffer == null) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(),
                  "There is an error getting the stylized face. It usually results from incorrect"
                      + " options of unsupported OutputType of given model.");
            }
            ByteBufferImageBuilder imageBuilder =
                new ByteBufferImageBuilder(imageBuffer, width, height, imageFormat);

            return FaceStylizerResult.create(
                Optional.of(imageBuilder.build()),
                BaseVisionTaskApi.generateResultTimestampMs(
                    RunningMode.IMAGE, packets.get(IMAGE_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    // Empty output image packets indicates that no face stylization is applied.
    handler.setHandleTimestampBoundChanges(true);
    stylizerOptions.resultListener().ifPresent(handler::setResultListener);
    stylizerOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<FaceStylizerOptions>builder()
                .setTaskName(FaceStylizer.class.getSimpleName())
                .setTaskRunningModeName(RunningMode.IMAGE.name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(stylizerOptions)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new FaceStylizer(
        runner, RunningMode.IMAGE, stylizerOptions.resultListener().isPresent());
  }

  /**
   * Constructor to initialize an {@link FaceStylizer} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private FaceStylizer(TaskRunner taskRunner, RunningMode runningMode, boolean hasResultListener) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
    this.hasResultListener = hasResultListener;
  }

  /**
   * Performs face stylization on the provided single image with default image processing options,
   * i.e. without any rotation applied. Only use this method when the {@link FaceStylizer} is
   * created with {@link RunningMode#IMAGE}.
   *
   * <p>{@link FaceStylizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link android.graphics.Bitmap.Config#ARGB_8888}
   * </ul>
   *
   * <p>The input image can be of any size. The output image is the stylized image with the most
   * visible face. The stylized output image size is the same as the model output size. When no face
   * is detected on the input image, returns {@code Optional.empty()}.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error. Or if {@link FaceStylizer} is created
   *     with a {@link ResultListener}.
   */
  public FaceStylizerResult stylize(MPImage image) {
    return stylize(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs face stylization on the provided single image. Only use this method when the {@link
   * FaceStylizer} is created with {@link RunningMode#IMAGE}.
   *
   * <p>{@link FaceStylizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link android.graphics.Bitmap.Config#ARGB_8888}
   * </ul>
   *
   * <p>The input image can be of any size. The output image is the stylized image with the most
   * visible face. The stylized output image size is the same as the model output size. When no face
   * is detected on the input image, returns {@code Optional.empty()}.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link FaceStylizer} is created
   *     with a {@link ResultListener}.
   */
  public FaceStylizerResult stylize(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    if (hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is provided in the FaceStylizerOptions, but this method will return an"
              + " ImageSegmentationResult.");
    }
    return (FaceStylizerResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs face stylization on the provided single image with default image processing options,
   * i.e. without any rotation applied, and provides zero-copied results via {@link ResultListener}
   * in {@link FaceStylizerOptions}. Only use this method when the {@link FaceStylizer} is created
   * with {@link RunningMode#IMAGE}.
   *
   * <p>{@link FaceStylizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link android.graphics.Bitmap.Config#ARGB_8888}
   * </ul>
   *
   * <p>The input image can be of any size. The output image is the stylized image with the most
   * visible face. The stylized output image size is the same as the model output size. When no face
   * is detected on the input image, returns {@code Optional.empty()}.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link FaceStylizer} is not
   *     created with {@link ResultListener} set in {@link FaceStylizerOptions}.
   */
  public void stylizeWithResultListener(MPImage image) {
    stylizeWithResultListener(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs face stylization on the provided single image, and provides zero-copied results via
   * {@link ResultListener} in {@link FaceStylizerOptions}. Only use this method when the {@link
   * FaceStylizer} is created with {@link RunningMode#IMAGE}.
   *
   * <p>{@link FaceStylizer} supports the following color space types:
   *
   * <ul>
   *   <li>{@link android.graphics.Bitmap.Config#ARGB_8888}
   * </ul>
   *
   * <p>The input image can be of any size. The output image is the stylized image with the most
   * visible face. The stylized output image size is the same as the model output size. When no face
   * is detected on the input image, returns {@code Optional.empty()}.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link FaceStylizer} is not
   *     created with {@link ResultListener} set in {@link FaceStylizerOptions}.
   */
  public void stylizeWithResultListener(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    if (!hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is not set in the FaceStylizerOptions, but this method expects a"
              + " ResultListener to process ImageSegmentationResult.");
    }
    TaskResult unused = processImageData(image, imageProcessingOptions);
  }

  /** Options for setting up an {@link FaceStylizer}. */
  @AutoValue
  public abstract static class FaceStylizerOptions extends TaskOptions {

    /** Builder for {@link FaceStylizerOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the face stylizer task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets an optional {@link ResultListener} to receive the stylization results when the graph
       * pipeline is done processing an image.
       */
      public abstract Builder setResultListener(ResultListener<FaceStylizerResult, MPImage> value);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract FaceStylizerOptions autoBuild();

      /** Builds the {@link FaceStylizerOptions} instance. */
      public final FaceStylizerOptions build() {
        return autoBuild();
      }
    }

    abstract BaseOptions baseOptions();

    abstract Optional<ResultListener<FaceStylizerResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_FaceStylizer_FaceStylizerOptions.Builder();
    }

    /** Converts an {@link FaceStylizerOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      FaceStylizerGraphOptionsProto.FaceStylizerGraphOptions taskOptions =
          FaceStylizerGraphOptionsProto.FaceStylizerGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build())
              .build();

      return CalculatorOptions.newBuilder()
          .setExtension(FaceStylizerGraphOptionsProto.FaceStylizerGraphOptions.ext, taskOptions)
          .build();
    }
  }
}
