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

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.TensorsToSegmentationCalculatorOptionsProto;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
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
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult;
import com.google.mediapipe.tasks.vision.imagesegmenter.proto.ImageSegmenterGraphOptionsProto;
import com.google.mediapipe.tasks.vision.imagesegmenter.proto.SegmenterOptionsProto;
import com.google.mediapipe.util.proto.ColorProto.Color;
import com.google.mediapipe.util.proto.RenderDataProto.RenderAnnotation;
import com.google.mediapipe.util.proto.RenderDataProto.RenderData;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Performs interactive segmentation on images.
 *
 * <p>Note that, in addition to the standard segmentation API {@link segment} that takes an input
 * image and returns the outputs, but involves deep copy of the returns, InteractiveSegmenter also
 * supports the callback API, {@link segmentWithResultListener}, which allows you to access the
 * outputs through zero copy. Set {@link ResultListener} in {@link InteractiveSegmenterOptions}
 * properly to use the callback API.
 *
 * <p>The API expects a TFLite model with,<a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>. The model
 * expects input with 4 channels, where the first 3 channels represent RGB image, and the last
 * channel represents the user's region of interest.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that image segmenter runs on.
 *       </ul>
 *   <li>Input roi {@link RegionOfInterest}
 *       <ul>
 *         <li>Region of interest based on user interaction.
 *       </ul>
 *   <li>Output ImageSegmenterResult {@link ImageSegmenterResult}
 *       <ul>
 *         <li>An ImageSegmenterResult containing segmented masks.
 *       </ul>
 * </ul>
 */
public final class InteractiveSegmenter extends BaseVisionTaskApi {
  private static final String TAG = InteractiveSegmenter.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String ROI_IN_STREAM_NAME = "roi_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "IMAGE:" + IMAGE_IN_STREAM_NAME,
              "ROI:" + ROI_IN_STREAM_NAME,
              "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME));
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
      "mediapipe.tasks.vision.interactive_segmenter.InteractiveSegmenterGraph";
  private static final String TENSORS_TO_SEGMENTATION_CALCULATOR_NAME =
      "mediapipe.tasks.TensorsToSegmentationCalculator";
  private boolean hasResultListener = false;
  private List<String> labels = new ArrayList<>();

  static {
    ProtoUtil.registerTypeName(RenderData.class, "mediapipe.RenderData");
  }

  /**
   * Creates an {@link InteractiveSegmenter} instance from an {@link InteractiveSegmenterOptions}.
   *
   * @param context an Android {@link Context}.
   * @param segmenterOptions an {@link InteractiveSegmenterOptions} instance.
   * @throws MediaPipeException if there is an error during {@link InteractiveSegmenter} creation.
   */
  public static InteractiveSegmenter createFromOptions(
      Context context, InteractiveSegmenterOptions segmenterOptions) {
    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<ImageSegmenterResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageSegmenterResult, MPImage>() {
          @Override
          public ImageSegmenterResult convertToTaskResult(List<Packet> packets)
              throws MediaPipeException {
            if (packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX).isEmpty()) {
              return ImageSegmenterResult.create(
                  Optional.empty(),
                  Optional.empty(),
                  packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX).getTimestamp());
            }
            List<MPImage> segmentedMasks = new ArrayList<>();
            int width = PacketGetter.getImageWidth(packets.get(SEGMENTATION_OUT_STREAM_INDEX));
            int height = PacketGetter.getImageHeight(packets.get(SEGMENTATION_OUT_STREAM_INDEX));
            int imageFormat =
                segmenterOptions.outputType()
                        == InteractiveSegmenterOptions.OutputType.CONFIDENCE_MASK
                    ? MPImage.IMAGE_FORMAT_VEC32F1
                    : MPImage.IMAGE_FORMAT_ALPHA;
            int imageListSize =
                PacketGetter.getImageListSize(packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX));
            ByteBuffer[] buffersArray = new ByteBuffer[imageListSize];
            // If resultListener is not provided, the resulted MPImage is deep copied from mediapipe
            // graph. If provided, the result MPImage is wrapping the mediapipe packet memory.
            if (!segmenterOptions.resultListener().isPresent()) {
              for (int i = 0; i < imageListSize; i++) {
                buffersArray[i] =
                    ByteBuffer.allocateDirect(
                        width * height * (imageFormat == MPImage.IMAGE_FORMAT_VEC32F1 ? 4 : 1));
              }
            }
            if (!PacketGetter.getImageList(
                packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX),
                buffersArray,
                !segmenterOptions.resultListener().isPresent())) {
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
                Optional.of(segmentedMasks),
                Optional.empty(),
                BaseVisionTaskApi.generateResultTimestampMs(
                    RunningMode.IMAGE, packets.get(GROUPED_SEGMENTATION_OUT_STREAM_INDEX)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(IMAGE_OUT_STREAM_INDEX)))
                .build();
          }
        });
    segmenterOptions.resultListener().ifPresent(handler::setResultListener);
    segmenterOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<InteractiveSegmenterOptions>builder()
                .setTaskName(InteractiveSegmenter.class.getSimpleName())
                .setTaskRunningModeName(RunningMode.IMAGE.name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(OUTPUT_STREAMS)
                .setTaskOptions(segmenterOptions)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    return new InteractiveSegmenter(runner, segmenterOptions.resultListener().isPresent());
  }

  /**
   * Constructor to initialize an {@link InteractiveSegmenter} from a {@link TaskRunner}.
   *
   * @param taskRunner a {@link TaskRunner}.
   */
  private InteractiveSegmenter(TaskRunner taskRunner, boolean hasResultListener) {
    super(taskRunner, RunningMode.IMAGE, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
    this.hasResultListener = hasResultListener;
    populateLabels();
  }

  /**
   * Populate the labelmap in TensorsToSegmentationCalculator to labels field.
   *
   * @throws MediaPipeException if there is an error during finding TensorsToSegmentationCalculator.
   */
  private void populateLabels() {
    CalculatorGraphConfig graphConfig = this.runner.getCalculatorGraphConfig();

    boolean foundTensorsToSegmentation = false;
    for (CalculatorGraphConfig.Node node : graphConfig.getNodeList()) {
      if (node.getName().contains(TENSORS_TO_SEGMENTATION_CALCULATOR_NAME)) {
        if (foundTensorsToSegmentation) {
          throw new MediaPipeException(
              MediaPipeException.StatusCode.INTERNAL.ordinal(),
              "The graph has more than one mediapipe.tasks.TensorsToSegmentationCalculator.");
        }
        foundTensorsToSegmentation = true;
        TensorsToSegmentationCalculatorOptionsProto.TensorsToSegmentationCalculatorOptions options =
            node.getOptions()
                .getExtension(
                    TensorsToSegmentationCalculatorOptionsProto
                        .TensorsToSegmentationCalculatorOptions.ext);
        for (int i = 0; i < options.getLabelItemsMap().size(); i++) {
          Long labelKey = Long.valueOf(i);
          if (!options.getLabelItemsMap().containsKey(labelKey)) {
            throw new MediaPipeException(
                MediaPipeException.StatusCode.INTERNAL.ordinal(),
                "The lablemap have no expected key: " + labelKey);
          }
          labels.add(options.getLabelItemsMap().get(labelKey).getName());
        }
      }
    }
  }

  /**
   * Performs segmentation on the provided single image with default image processing options, given
   * user's region-of-interest, i.e. without any rotation applied. TODO update java doc
   * for input image format.
   *
   * <p>Users can represent user interaction through {@link RegionOfInterest}, which gives a hint to
   * perform segmentation focusing on the given region of interest.
   *
   * <p>{@link InteractiveSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @throws MediaPipeException if there is an internal error. Or if {@link InteractiveSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segment(MPImage image, RegionOfInterest roi) {
    return segment(image, roi, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs segmentation on the provided single image, given user's region-of-interest.
   * TODO update java doc for input image format.
   *
   * <p>Users can represent user interaction through {@link RegionOfInterest}, which gives a hint to
   * perform segmentation focusing on the given region of interest.
   *
   * <p>{@link InteractiveSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link InteractiveSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segment(
      MPImage image, RegionOfInterest roi, ImageProcessingOptions imageProcessingOptions) {
    if (hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is provided in the InteractiveSegmenterOptions, but this method will"
              + " return an ImageSegmentationResult.");
    }
    validateImageProcessingOptions(imageProcessingOptions);
    return processImageWithRoi(image, roi, imageProcessingOptions);
  }

  /**
   * Performs segmentation on the provided single image with default image processing options, given
   * user's region-of-interest, i.e. without any rotation applied, and provides zero-copied results
   * via {@link ResultListener} in {@link InteractiveSegmenterOptions}.
   *
   * <p>TODO update java doc for input image format.
   *
   * <p>Users can represent user interaction through {@link RegionOfInterest}, which gives a hint to
   * perform segmentation focusing on the given region of interest.
   *
   * <p>{@link InteractiveSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link InteractiveSegmenter} is
   *     not created with {@link ResultListener} set in {@link InteractiveSegmenterOptions}.
   */
  public void segmentWithResultListener(MPImage image, RegionOfInterest roi) {
    segmentWithResultListener(image, roi, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs segmentation on the provided single image given user's region-of-interest, and
   * provides zero-copied results via {@link ResultListener} in {@link InteractiveSegmenterOptions}.
   *
   * <p>TODO update java doc for input image format.
   *
   * <p>Users can represent user interaction through {@link RegionOfInterest}, which gives a hint to
   * perform segmentation focusing on the given region of interest.
   *
   * <p>{@link InteractiveSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link InteractiveSegmenter} is
   *     not created with {@link ResultListener} set in {@link InteractiveSegmenterOptions}.
   */
  public void segmentWithResultListener(
      MPImage image, RegionOfInterest roi, ImageProcessingOptions imageProcessingOptions) {
    if (!hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is not set in the InteractiveSegmenterOptions, but this method expects a"
              + " ResultListener to process ImageSegmentationResult.");
    }
    validateImageProcessingOptions(imageProcessingOptions);
    ImageSegmenterResult unused = processImageWithRoi(image, roi, imageProcessingOptions);
  }

  /**
   * Get the category label list of the ImageSegmenter can recognize. For CATEGORY_MASK type, the
   * index in the category mask corresponds to the category in the label list. For CONFIDENCE_MASK
   * type, the output mask list at index corresponds to the category in the label list.
   *
   * <p>If there is no labelmap provided in the model file, empty label list is returned.
   */
  List<String> getLabels() {
    return labels;
  }

  /** Options for setting up an {@link InteractiveSegmenter}. */
  @AutoValue
  public abstract static class InteractiveSegmenterOptions extends TaskOptions {

    /** Builder for {@link InteractiveSegmenterOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the base options for the image segmenter task. */
      public abstract Builder setBaseOptions(BaseOptions value);

      /** The output type from image segmenter. */
      public abstract Builder setOutputType(OutputType value);

      /**
       * Sets an optional {@link ResultListener} to receive the segmentation results when the graph
       * pipeline is done processing an image.
       */
      public abstract Builder setResultListener(
          ResultListener<ImageSegmenterResult, MPImage> value);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract InteractiveSegmenterOptions autoBuild();

      /** Builds the {@link InteractiveSegmenterOptions} instance. */
      public final InteractiveSegmenterOptions build() {
        return autoBuild();
      }
    }

    abstract BaseOptions baseOptions();

    abstract OutputType outputType();

    abstract Optional<ResultListener<ImageSegmenterResult, MPImage>> resultListener();

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
      return new AutoValue_InteractiveSegmenter_InteractiveSegmenterOptions.Builder()
          .setOutputType(OutputType.CATEGORY_MASK);
    }

    /**
     * Converts an {@link InteractiveSegmenterOptions} to a {@link CalculatorOptions} protobuf
     * message.
     */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.Builder taskOptionsBuilder =
          ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.newBuilder()
              .setBaseOptions(
                  BaseOptionsProto.BaseOptions.newBuilder()
                      .setUseStreamMode(false)
                      .mergeFrom(convertBaseOptionsToProto(baseOptions()))
                      .build());

      SegmenterOptionsProto.SegmenterOptions.Builder segmenterOptionsBuilder =
          SegmenterOptionsProto.SegmenterOptions.newBuilder();
      if (outputType() == OutputType.CONFIDENCE_MASK) {
        segmenterOptionsBuilder.setOutputType(
            SegmenterOptionsProto.SegmenterOptions.OutputType.CONFIDENCE_MASK);
      } else if (outputType() == OutputType.CATEGORY_MASK) {
        segmenterOptionsBuilder.setOutputType(
            SegmenterOptionsProto.SegmenterOptions.OutputType.CATEGORY_MASK);
      }

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
      throw new IllegalArgumentException(
          "InteractiveSegmenter doesn't support region-of-interest.");
    }
  }

  /** The Region-Of-Interest (ROI) to interact with. */
  public static class RegionOfInterest {
    private NormalizedKeypoint keypoint;

    private RegionOfInterest() {}

    /**
     * Creates a {@link RegionOfInterest} instance representing a single point pointing to the
     * object that the user wants to segment.
     */
    public static RegionOfInterest create(NormalizedKeypoint keypoint) {
      RegionOfInterest roi = new RegionOfInterest();
      roi.keypoint = keypoint;
      return roi;
    }
  }

  /**
   * Converts a {@link RegionOfInterest} instance into a {@link RenderData} protobuf message
   *
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @throws IllegalArgumentException if {@link RegionOfInterest} does not represent a valid user
   *     interaction.
   */
  private static RenderData convertToRenderData(RegionOfInterest roi) {
    RenderData.Builder builder = RenderData.newBuilder();
    if (roi.keypoint != null) {
      return builder
          .addRenderAnnotations(
              RenderAnnotation.newBuilder()
                  .setColor(Color.newBuilder().setR(255))
                  .setPoint(
                      RenderAnnotation.Point.newBuilder()
                          .setX(roi.keypoint.x())
                          .setY(roi.keypoint.y())))
          .build();
    }

    throw new IllegalArgumentException(
        "RegionOfInterest does not include a valid user interaction");
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * <p>This is almost the same as {@link BaseVisionTaskApi.processImageData} except accepting an
   * additional {@link RegionOfInterest}.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param roi a {@link RegionOfInterest} object to represent user interaction.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @throws MediaPipeException if the task is not in the image mode.
   */
  private ImageSegmenterResult processImageWithRoi(
      MPImage image, RegionOfInterest roi, ImageProcessingOptions imageProcessingOptions) {
    if (runningMode != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the image mode. Current running mode:"
              + runningMode.name());
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(IMAGE_IN_STREAM_NAME, runner.getPacketCreator().createImage(image));
    RenderData renderData = convertToRenderData(roi);
    inputPackets.put(ROI_IN_STREAM_NAME, runner.getPacketCreator().createProto(renderData));
    inputPackets.put(
        NORM_RECT_IN_STREAM_NAME,
        runner
            .getPacketCreator()
            .createProto(convertToNormalizedRect(imageProcessingOptions, image)));
    return (ImageSegmenterResult) runner.process(inputPackets);
  }
}
