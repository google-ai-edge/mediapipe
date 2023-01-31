// Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

package com.google.mediapipe.tasks.vision.imagesegmenter;

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
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imagesegmenter.proto.ImageSegmenterGraphOptionsProto;
import com.google.mediapipe.tasks.vision.imagesegmenter.proto.SegmenterOptionsProto;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * Performs image segmentation on images.
 *
 * <p>Note that, unlike other vision tasks, the output of ImageSegmenter is provided through a
 * user-defined callback function even for the synchronous API. This makes it possible for
 * ImageSegmenter to return the output masks without any copy. {@link ResultListener} must be set in
 * the {@link ImageSegmenterOptions} for all {@link RunningMode}.
 *
 * <p>The API expects a TFLite model with,<a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that image segmenter runs on.
 *       </ul>
 *   <li>Output ImageSegmenterResult {@link ImageSgmenterResult}
 *       <ul>
 *         <li>An ImageSegmenterResult containing segmented masks.
 *       </ul>
 * </ul>
 */
public final class ImageSegmenter extends BaseVisionTaskApi {
  private static final String TAG = ImageSegmenter.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList("IMAGE:" + IMAGE_IN_STREAM_NAME, "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));
  private static final List<String> OUTPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "GROUPED_SEGMENTATION:segmented_mask_out",
              "IMAGE:image_out",
              "SEGMENTATION:0:segmentation"));
  private static final int GROUPED_SEGMENTATION_OUT_STREAM_INDEX = 0;
  private static final int IMAGE_OUT_STREAM_INDEX = 1;
  private static final int SEGMENTATION_OUT_STREAM_INDEX = 2;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph";

  /**
   * Creates an {@link ImageSegmenter} instance from an {@link ImageSegmenterOptions}.
   *
   * @param context an Android {@link Context}.
   * @param segmenterOptions an {@link ImageSegmenterOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageSegmenter} creation.
   */
  public static ImageSegmenter createFromOptions(
      Context context, ImageSegmenterOptions segmenterOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<ImageSegmenterResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageSegmenterResult, MPImage>() {
          @Override
          public ImageSegmenterResult convertToTaskResult(List<Packet> packets)
              throws MediaPipeException {
            if (packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX).isEmpty()) {
              return ImageSegmenterResult.create(
                  new ArrayList<>(),
                  packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX).getTimestamp());
            }
            List<MPImage> segmentedMasks = new ArrayList<>();
            int width = PacketGetter.getImageWidth(packets.get(SEGMENTATION_OUT_STREAM_INDEX));
            int height = PacketGetter.getImageHeight(packets.get(SEGMENTATION_OUT_STREAM_INDEX));
            int imageFormat =
                segmenterOptions.outputType() == ImageSegmenterOptions.OutputType.CONFIDENCE_MASK
                    ? MPImage.IMAGE_FORMAT_VEC32F1
                    : MPImage.IMAGE_FORMAT_ALPHA;
            int imageListSize =
                PacketGetter.getImageListSize(packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX));
            ByteBuffer[] buffersArray = new ByteBuffer[imageListSize];
            if (!PacketGetter.getImageList(
                packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX), buffersArray, false)) {
              throw new MediaPipeException(
                  MediaPipeException.StatusCode.INTERNAL.ordinal(),
                  "There is an error getting segmented masks. It usually results from incorrect"
                      + " options of unsupported OutputType of given model.");
            }
            for (ByteBuffer buffer : buffersArray) {
              ByteBufferImageBuilder builder =
                  new ByteBufferImageBuilder(buffer, width, height, imageFormat);
              segmentedMasks.add(builder.build());
            }

            return ImageSegmenterResult.create(
                segmentedMasks,
                BaseVisionTaskApi.generateResultTimestampMs(
                    segmenterOptions.runningMode(),
                    packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    handler.setResultListener(segmenterOptions.resultListener());
    segmenterOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ImageSegmenterOptions>builder()
                .setTaskName(ImageSegmenter.class.getSimpleName())
                .setTaskRunningModeName(segmenterOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(segmenterOptions)
                .setEnableFlowLimiting(segmenterOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new ImageSegmenter(runner, segmenterOptions.runningMode());
  }

  /**
   * Constructor to initialize an {@link ImageSegmenter} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private ImageSegmenter(TaskRunner taskRunner, RunningMode runningMode) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
  }

  /**
   * Performs image segmentation on the provided single image with default image processing options,
   * i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method when the
   * {@link ImageSegmenter} is created with {@link RunningMode.IMAGE}. TODO update java
   * doc for input image format.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public void segment(MPImage image) {
    segment(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs image segmentation on the provided single image, and the results will be available via
   * the {@link ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method
   * when the {@link ImageSegmenter} is created with {@link RunningMode.IMAGE}. TODO
   * update java doc for input image format.
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
  public void segment(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    validateImageProcessingOptions(imageProcessingOptions);
    ImageSegmenterResult unused =
        (ImageSegmenterResult) processImageData(image, imageProcessingOptions);
  }

  /**
   * Performs image segmentation on the provided video frame with default image processing options,
   * i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method when the
   * {@link HandLandmarker} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void segmentForVideo(MPImage image, long timestampMs) {
    segmentForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame, and the results will be available via
   * the {@link ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method
   * when the {@link ImageSegmenter} is created with {@link RunningMode.VIDEO}.
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
  public void segmentForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    ImageSegmenterResult unused =
        (ImageSegmenterResult) processVideoData(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform hand landmarks detection with default image processing
   * options, i.e. without any rotation applied, and the results will be available via the {@link
   * ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method when the
   * {@link ImageSegmenter } is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the image segmenter. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void segmentAsync(MPImage image, long timestampMs) {
    segmentAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform image segmentation, and the results will be available via the
   * {@link ResultListener} provided in the {@link ImageSegmenterOptions}. Only use this method when
   * the {@link ImageSegmenter} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the image segmenter. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
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
  public void segmentAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    validateImageProcessingOptions(imageProcessingOptions);
    sendLiveStreamData(image, imageProcessingOptions, timestampMs);
  }

  /** Options for setting up an {@link ImageSegmenter}. */
  @AutoValue
  public abstract static class ImageSegmenterOptions extends TaskOptions {

    /** Builder for {@link ImageSegmenterOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the image segmenter task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /**
       * Sets the running mode for the image segmenter task. Default to the image mode. Image
       * segmenter has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for segmenting image on single image inputs.
       *   <li>VIDEO: The mode for segmenting image on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for segmenting image on a live stream of input data, such
       *       as from camera. In this mode, {@code setResultListener} must be called to set up a
       *       listener to receive the recognition results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode value);

      /**
       * The locale to use for display names specified through the TFLite Model Metadata, if any.
       * Defaults to English.
       */
      public abstract Builder setDisplayNamesLocale(String value);

      /** The output type from image segmenter. */
      public abstract Builder setOutputType(OutputType value);

      /**
       * Sets the {@link ResultListener} to receive the segmentation results when the graph pipeline
       * is done processing an image.
       */
      public abstract Builder setResultListener(
          ResultListener<ImageSegmenterResult, MPImage> value);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract ImageSegmenterOptions autoBuild();

      /**
       * Validates and builds the {@link ImageSegmenterOptions} instance.
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the image segmenter is
       *     in the live stream mode.
       */
      public final ImageSegmenterOptions build() {
        ImageSegmenterOptions options = autoBuild();
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract String displayNamesLocale();

    abstract OutputType outputType();

    abstract ResultListener<ImageSegmenterResult, MPImage> resultListener();

    abstract Optional<ErrorListener> errorListener();

    /** The output type of segmentation results. */
    public enum OutputType {
      // Gives a single output mask where each pixel represents the class which
      // the pixel in the original image was predicted to belong to.
      CATEGORY_MASK,
      // Gives a list of output masks where, for each mask, each pixel represents
      // the prediction confidence, usually in the [0, 1] range.
      CONFIDENCE_MASK
    }

    public static Builder builder() {
      return new AutoValue_ImageSegmenter_ImageSegmenterOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setDisplayNamesLocale("en")
          .setOutputType(OutputType.CATEGORY_MASK)
          .setResultListener((result, image) -> {});
    }

    /**
     * Converts an {@link ImageSegmenterOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.Builder taskOptionsBuilder =
          ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(runningMode() != RunningMode.IMAGE)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build())
              .setDisplayNamesLocale(displayNamesLocale());

      SegmenterOptionsProto.SegmenterOptions.Builder segmenterOptionsBuilder =
          SegmenterOptionsProto.SegmenterOptions.newBuilder();
      if (outputType() == OutputType.CONFIDENCE_MASK) {
        segmenterOptionsBuilder.setOutputType(
            SegmenterOptionsProto.SegmenterOptions.OutputType.CONFIDENCE_MASK);
      } else if (outputType() == OutputType.CATEGORY_MASK) {
        segmenterOptionsBuilder.setOutputType(
            SegmenterOptionsProto.SegmenterOptions.OutputType.CATEGORY_MASK);
      }
      // TODO: remove this once activation is handled in metadata and grpah level.
      segmenterOptionsBuilder.setActivation(
          SegmenterOptionsProto.SegmenterOptions.Activation.SOFTMAX);
      taskOptionsBuilder.setSegmenterOptions(segmenterOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.ext,
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
      throw new IllegalArgumentException("ImageSegmenter doesn't support region-of-interest.");
    }
  }
}
