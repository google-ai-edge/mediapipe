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

package com.google.mediapipe.tasks.vision.imagesegmenter;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.TensorsToSegmentationCalculatorOptionsProto;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;

/**
 * Performs image segmentation on images.
 *
 * <p>Note that, in addition to the standard segmentation API, {@link segment} and {@link
 * segmentForVideo}, that take an input image and return the outputs, but involves deep copy of the
 * returns, ImageSegmenter also supports the callback API, {@link segmentWithResultListener} and
 * {@link segmentForVideoWithResultListener}, which allow you to access the outputs through zero
 * copy.
 *
 * <p>The callback API is available for all {@link RunningMode} in ImageSegmenter. Set {@link
 * ResultListener} in {@link ImageSegmenterOptions} properly to use the callback API.
 *
 * <p>The API expects a TFLite model with,<a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <ul>
 *   <li>Input image {@link MPImage}
 *       <ul>
 *         <li>The image that image segmenter runs on.
 *       </ul>
 *   <li>Output ImageSegmenterResult {@link ImageSegmenterResult}
 *       <ul>
 *         <li>An ImageSegmenterResult containing segmented masks.
 *       </ul>
 * </ul>
 */
public final class ImageSegmenter extends BaseVisionTaskApi {
  private static final String TAG = ImageSegmenter.class.getSimpleName();
  private static final String IMAGE_IN_STREAM_NAME = "image_in";
  private static final String NORM_RECT_IN_STREAM_NAME = "norm_rect_in";
  private static final String OUTPUT_SIZE_IN_STREAM_NAME = "output_size_in";
  private static final List<String> INPUT_STREAMS =
      Collections.unmodifiableList(
          Arrays.asList(
              "IMAGE:" + IMAGE_IN_STREAM_NAME,
              "NORM_RECT:" + NORM_RECT_IN_STREAM_NAME,
              "OUTPUT_SIZE:" + OUTPUT_SIZE_IN_STREAM_NAME));
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.image_segmenter.ImageSegmenterGraph";
  private static final String TENSORS_TO_SEGMENTATION_CALCULATOR_NAME =
      "mediapipe.tasks.TensorsToSegmentationCalculator";
  private boolean hasResultListener = false;
  private List<String> labels = new ArrayList<>();

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Creates an {@link ImageSegmenter} instance from an {@link ImageSegmenterOptions}.
   *
   * @param context an Android {@link Context}.
   * @param segmenterOptions an {@link ImageSegmenterOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageSegmenter} creation.
   */
  public static ImageSegmenter createFromOptions(
      Context context, ImageSegmenterOptions segmenterOptions) {
    if (!segmenterOptions.outputConfidenceMasks() && !segmenterOptions.outputCategoryMask()) {
      throw new IllegalArgumentException(
          "At least one of `outputConfidenceMasks` and `outputCategoryMask` must be set.");
    }
    List<String> outputStreams = new ArrayList<>();
    // Add an output stream to the output stream list, and get the added output stream index.
    BiFunction<List<String>, String, Integer> getStreamIndex =
        (List<String> streams, String streamName) -> {
          streams.add(streamName);
          return streams.size() - 1;
        };
    final int confidenceMasksOutStreamIndex =
        segmenterOptions.outputConfidenceMasks()
            ? getStreamIndex.apply(outputStreams, "CONFIDENCE_MASKS:confidence_masks")
            : -1;
    final int categoryMaskOutStreamIndex =
        segmenterOptions.outputCategoryMask()
            ? getStreamIndex.apply(outputStreams, "CATEGORY_MASK:category_mask")
            : -1;
    final int qualityScoresOutStreamIndex =
        getStreamIndex.apply(outputStreams, "QUALITY_SCORES:quality_scores");
    final int imageOutStreamIndex = getStreamIndex.apply(outputStreams, "IMAGE:image_out");

    // TODO: Consolidate OutputHandler and TaskRunner.
    OutputHandler<ImageSegmenterResult, MPImage> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageSegmenterResult, MPImage>() {
          @Override
          public ImageSegmenterResult convertToTaskResult(List<Packet> packets)
              throws MediaPipeException {
            if (packets.get(imageOutStreamIndex).isEmpty()) {
              return ImageSegmenterResult.create(
                  Optional.empty(),
                  Optional.empty(),
                  new ArrayList<>(),
                  packets.get(imageOutStreamIndex).getTimestamp());
            }
            boolean copyImage = !segmenterOptions.resultListener().isPresent();
            Optional<List<MPImage>> confidenceMasks = Optional.empty();
            if (segmenterOptions.outputConfidenceMasks()) {
              int width =
                  PacketGetter.getImageWidthFromImageList(
                      packets.get(confidenceMasksOutStreamIndex));
              int height =
                  PacketGetter.getImageHeightFromImageList(
                      packets.get(confidenceMasksOutStreamIndex));
              confidenceMasks = Optional.of(new ArrayList<MPImage>());
              int confidenceMasksListSize =
                  PacketGetter.getImageListSize(packets.get(confidenceMasksOutStreamIndex));
              ByteBuffer[] buffersArray = new ByteBuffer[confidenceMasksListSize];
              // If resultListener is not provided, the resulted MPImage is deep copied from
              // mediapipe graph. If provided, the result MPImage is wrapping the mediapipe packet
              // memory.
              if (copyImage) {
                for (int i = 0; i < confidenceMasksListSize; i++) {
                  buffersArray[i] = ByteBuffer.allocateDirect(width * height * 4);
                }
              }
              if (!PacketGetter.getImageList(
                  packets.get(confidenceMasksOutStreamIndex), buffersArray, copyImage)) {
                throw new MediaPipeException(
                    MediaPipeException.StatusCode.INTERNAL.ordinal(),
                    "There is an error getting confidence masks.");
              }
              for (ByteBuffer buffer : buffersArray) {
                ByteBufferImageBuilder builder =
                    new ByteBufferImageBuilder(buffer, width, height, MPImage.IMAGE_FORMAT_VEC32F1);
                confidenceMasks.get().add(builder.build());
              }
            }
            Optional<MPImage> categoryMask = Optional.empty();
            if (segmenterOptions.outputCategoryMask()) {
              int width = PacketGetter.getImageWidth(packets.get(categoryMaskOutStreamIndex));
              int height = PacketGetter.getImageHeight(packets.get(categoryMaskOutStreamIndex));
              ByteBuffer buffer;
              if (copyImage) {
                buffer = ByteBuffer.allocateDirect(width * height);
                if (!PacketGetter.getImageData(packets.get(categoryMaskOutStreamIndex), buffer)) {
                  throw new MediaPipeException(
                      MediaPipeException.StatusCode.INTERNAL.ordinal(),
                      "There is an error getting category mask.");
                }
              } else {
                buffer = PacketGetter.getImageDataDirectly(packets.get(categoryMaskOutStreamIndex));
              }
              ByteBufferImageBuilder builder =
                  new ByteBufferImageBuilder(buffer, width, height, MPImage.IMAGE_FORMAT_ALPHA);
              categoryMask = Optional.of(builder.build());
            }
            float[] qualityScores =
                PacketGetter.getFloat32Vector(packets.get(qualityScoresOutStreamIndex));
            List<Float> qualityScoresList = new ArrayList<>(qualityScores.length);
            for (float score : qualityScores) {
              qualityScoresList.add(score);
            }
            return ImageSegmenterResult.create(
                confidenceMasks,
                categoryMask,
                qualityScoresList,
                BaseVisionTaskApi.generateResultTimestampMs(
                    segmenterOptions.runningMode(), packets.get(imageOutStreamIndex)));
          }

          @Override
          public MPImage convertToTaskInput(List<Packet> packets) {
            return new BitmapImageBuilder(
                    AndroidPacketGetter.getBitmapFromRgb(packets.get(imageOutStreamIndex)))
                .build();
          }
        });
    segmenterOptions.resultListener().ifPresent(handler::setResultListener);
    segmenterOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ImageSegmenterOptions>builder()
                .setTaskName(ImageSegmenter.class.getSimpleName())
                .setTaskRunningModeName(segmenterOptions.runningMode().name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(INPUT_STREAMS)
                .setOutputStreams(outputStreams)
                .setTaskOptions(segmenterOptions)
                .setEnableFlowLimiting(segmenterOptions.runningMode() == RunningMode.LIVE_STREAM)
                .build(),
            handler);
    return new ImageSegmenter(
        runner, segmenterOptions.runningMode(), segmenterOptions.resultListener().isPresent());
  }

  /**
   * Constructor to initialize an {@link ImageSegmenter} from a {@link TaskRunner} and a {@link
   * RunningMode}.
   *
   * @param taskRunner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  private ImageSegmenter(
      TaskRunner taskRunner, RunningMode runningMode, boolean hasResultListener) {
    super(taskRunner, runningMode, IMAGE_IN_STREAM_NAME, NORM_RECT_IN_STREAM_NAME);
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
   * Performs image segmentation on the provided single image with default image processing options,
   * i.e. without any rotation applied. The output mask has the same size as the input image. Only
   * use this method when the {@link ImageSegmenter} is created with {@link RunningMode.IMAGE}.
   * TODO update java doc for input image format.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segment(MPImage image) {
    return segment(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs image segmentation on the provided single image. The output mask has the same size as
   * the input image. Only use this method when the {@link ImageSegmenter} is created with {@link
   * RunningMode.IMAGE}. TODO update java doc for input image format.
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
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segment(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    return segment(
        image,
        SegmentationOptions.builder()
            .setOutputWidth(image.getWidth())
            .setOutputHeight(image.getHeight())
            .setImageProcessingOptions(imageProcessingOptions)
            .build());
  }

  /**
   * Performs image segmentation on the provided single image. Only use this method when the {@link
   * ImageSegmenter} is created with {@link RunningMode.IMAGE}. TODO update java doc
   * for input image format.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param segmentationOptions the {@link SegmentationOptions} used to configure the runtime
   *     behavior of the {@link ImageSegmenter}.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segment(MPImage image, SegmentationOptions segmentationOptions) {
    if (hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is provided in the ImageSegmenterOptions, but this method will return an"
              + " ImageSegmentationResult.");
    }
    return (ImageSegmenterResult) processImageData(buildInputPackets(image, segmentationOptions));
  }

  /**
   * Performs image segmentation on the provided single image with default image processing options,
   * i.e. without any rotation applied, and provides zero-copied results via {@link ResultListener}
   * in {@link ImageSegmenterOptions}. The output mask has the same size as the input image. Only
   * use this method when the {@link ImageSegmenter} is created with {@link RunningMode.IMAGE}.
   *
   * <p>TODO update java doc for input image format.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentWithResultListener(MPImage image) {
    segmentWithResultListener(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs image segmentation on the provided single image, and provides zero-copied results via
   * {@link ResultListener} in {@link ImageSegmenterOptions}. The output mask has the same size as
   * the input image. Only use this method when the {@link ImageSegmenter} is created with {@link
   * RunningMode.IMAGE}.
   *
   * <p>TODO update java doc for input image format.
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
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentWithResultListener(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    segmentWithResultListener(
        image,
        SegmentationOptions.builder()
            .setOutputWidth(image.getWidth())
            .setOutputHeight(image.getHeight())
            .setImageProcessingOptions(imageProcessingOptions)
            .build());
  }

  /**
   * Performs image segmentation on the provided single image, and provides zero-copied results via
   * {@link ResultListener} in {@link ImageSegmenterOptions}. Only use this method when the {@link
   * ImageSegmenter} is created with {@link RunningMode.IMAGE}.
   *
   * <p>TODO update java doc for input image format.
   *
   * <p>{@link ImageSegmenter} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param segmentationOptions the {@link SegmentationOptions} used to configure the runtime
   *     behavior of the {@link ImageSegmenter}.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentWithResultListener(MPImage image, SegmentationOptions segmentationOptions) {
    if (!hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is not set in the ImageSegmenterOptions, but this method expects a"
              + " ResultListener to process ImageSegmentationResult.");
    }
    ImageSegmenterResult unused =
        (ImageSegmenterResult) processImageData(buildInputPackets(image, segmentationOptions));
  }

  /**
   * Performs image segmentation on the provided video frame with default image processing options,
   * i.e. without any rotation applied. The output mask has the same size as the input image. Only
   * use this method when the {@link ImageSegmenter} is created with {@link RunningMode.VIDEO}.
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
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segmentForVideo(MPImage image, long timestampMs) {
    return segmentForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame. The output mask has the same size as
   * the input image. Only use this method when the {@link ImageSegmenter} is created with {@link
   * RunningMode.VIDEO}.
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
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference. Note that region-of-interest is <b>not</b> supported
   *     by this task: specifying {@link ImageProcessingOptions#regionOfInterest()} will result in
   *     this method throwing an IllegalArgumentException.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segmentForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    return segmentForVideo(
        image,
        SegmentationOptions.builder()
            .setOutputWidth(image.getWidth())
            .setOutputHeight(image.getHeight())
            .setImageProcessingOptions(imageProcessingOptions)
            .build(),
        timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame. Only use this method when the {@link
   * ImageSegmenter} is created with {@link RunningMode.VIDEO}.
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
   * @param segmentationOptions the {@link SegmentationOptions} used to configure the runtime
   *     behavior of the {@link ImageSegmenter}.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is
   *     created with a {@link ResultListener}.
   */
  public ImageSegmenterResult segmentForVideo(
      MPImage image, SegmentationOptions segmentationOptions, long timestampMs) {
    if (hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is provided in the ImageSegmenterOptions, but this method will return an"
              + " ImageSegmentationResult.");
    }
    return (ImageSegmenterResult)
        processVideoData(buildInputPackets(image, segmentationOptions), timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame with default image processing options,
   * i.e. without any rotation applied, and provides zero-copied results via {@link ResultListener}
   * in {@link ImageSegmenterOptions}. The output mask has the same size as the input image. Only
   * use this method when the {@link ImageSegmenter} is created with {@link RunningMode.VIDEO}.
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
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentForVideoWithResultListener(MPImage image, long timestampMs) {
    segmentForVideoWithResultListener(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame, and provides zero-copied results via
   * {@link ResultListener} in {@link ImageSegmenterOptions}. The output mask has the same size as
   * the input image. Only use this method when the {@link ImageSegmenter} is created with {@link
   * RunningMode.VIDEO}.
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
   * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
   *     region-of-interest.
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentForVideoWithResultListener(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    segmentForVideoWithResultListener(
        image,
        SegmentationOptions.builder()
            .setOutputWidth(image.getWidth())
            .setOutputHeight(image.getHeight())
            .setImageProcessingOptions(imageProcessingOptions)
            .build(),
        timestampMs);
  }

  /**
   * Performs image segmentation on the provided video frame, and provides zero-copied results via
   * {@link ResultListener} in {@link ImageSegmenterOptions}. Only use this method when the {@link
   * ImageSegmenter} is created with {@link RunningMode.VIDEO}.
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
   * @param segmentationOptions the {@link SegmentationOptions} used to configure the runtime
   *     behavior of the {@link ImageSegmenter}.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error. Or if {@link ImageSegmenter} is not
   *     created with {@link ResultListener} set in {@link ImageSegmenterOptions}.
   */
  public void segmentForVideoWithResultListener(
      MPImage image, SegmentationOptions segmentationOptions, long timestampMs) {
    if (!hasResultListener) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "ResultListener is not set in the ImageSegmenterOptions, but this method expects a"
              + " ResultListener to process ImageSegmentationResult.");
    }
    ImageSegmenterResult unused =
        (ImageSegmenterResult)
            processVideoData(buildInputPackets(image, segmentationOptions), timestampMs);
  }

  /**
   * Sends live image data to perform image segmentation with default image processing options, i.e.
   * without any rotation applied, and the results will be available via the {@link ResultListener}
   * provided in the {@link ImageSegmenterOptions}. The output mask has the same size as the input
   * image. Only use this method when the {@link ImageSegmenter } is created with {@link
   * RunningMode.LIVE_STREAM}.
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
   * {@link ResultListener} provided in the {@link ImageSegmenterOptions}. The output mask has the
   * same size as the input image. Only use this method when the {@link ImageSegmenter} is created
   * with {@link RunningMode.LIVE_STREAM}.
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
    segmentAsync(
        image,
        SegmentationOptions.builder()
            .setOutputWidth(image.getWidth())
            .setOutputHeight(image.getHeight())
            .setImageProcessingOptions(imageProcessingOptions)
            .build(),
        timestampMs);
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
   * @param segmentationOptions the {@link SegmentationOptions} used to configure the runtime
   *     behavior of the {@link ImageSegmenter}.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void segmentAsync(
      MPImage image, SegmentationOptions segmentationOptions, long timestampMs) {
    sendLiveStreamData(buildInputPackets(image, segmentationOptions), timestampMs);
  }

  /**
   * Get the category label list of the ImageSegmenter can recognize. For CATEGORY_MASK type, the
   * index in the category mask corresponds to the category in the label list. For CONFIDENCE_MASK
   * type, the output mask list at index corresponds to the category in the label list.
   *
   * <p>If there is no labelmap provided in the model file, empty label list is returned.
   */
  public List<String> getLabels() {
    return labels;
  }

  /** Options for configuring runtime behavior of {@link ImageSegmenter}. */
  @AutoValue
  public abstract static class SegmentationOptions {

    /** Builder for {@link SegmentationOptions} */
    @AutoValue.Builder
    public abstract static class Builder {

      /** Set the width of the output segmentation masks. */
      public abstract Builder setOutputWidth(int value);

      /** Set the height of the output segmentation masks. */
      public abstract Builder setOutputHeight(int value);

      /** Set the image processing options. */
      public abstract Builder setImageProcessingOptions(ImageProcessingOptions value);

      abstract SegmentationOptions autoBuild();

      /**
       * Validates and builds the {@link SegmentationOptions} instance.
       *
       * @throws IllegalArgumentException if the {@link ImageProcessingOptions} specify a
       *     region-of-interest.
       */
      public final SegmentationOptions build() {
        SegmentationOptions options = autoBuild();
        if (options.outputWidth() <= 0 || options.outputHeight() <= 0) {
          throw new IllegalArgumentException(
              "Both outputWidth and outputHeight must be larger than 0.");
        }
        if (options.imageProcessingOptions().regionOfInterest().isPresent()) {
          throw new IllegalArgumentException("ImageSegmenter doesn't support region-of-interest.");
        }
        return options;
      }
    }

    abstract int outputWidth();

    abstract int outputHeight();

    abstract ImageProcessingOptions imageProcessingOptions();

    public static Builder builder() {
      return new AutoValue_ImageSegmenter_SegmentationOptions.Builder()
          .setImageProcessingOptions(ImageProcessingOptions.builder().build());
    }
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

      /** Whether to output confidence masks. */
      public abstract Builder setOutputConfidenceMasks(boolean value);

      /** Whether to output category mask. */
      public abstract Builder setOutputCategoryMask(boolean value);

      /**
       * /** Sets an optional {@link ResultListener} to receive the segmentation results when the
       * graph pipeline is done processing an image.
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
       *     properly configured. The result listener must be set when the image segmenter is in the
       *     live stream mode.
       */
      public final ImageSegmenterOptions build() {
        ImageSegmenterOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The image segmenter is in the live stream mode, a user-defined result listener"
                    + " must be provided in ImageSegmenterOptions.");
          }
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract String displayNamesLocale();

    abstract boolean outputConfidenceMasks();

    abstract boolean outputCategoryMask();

    abstract Optional<ResultListener<ImageSegmenterResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    public static Builder builder() {
      return new AutoValue_ImageSegmenter_ImageSegmenterOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setDisplayNamesLocale("en")
          .setOutputConfidenceMasks(true)
          .setOutputCategoryMask(false);
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
      taskOptionsBuilder.setSegmenterOptions(segmenterOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }

  private Map<String, Packet> buildInputPackets(
      MPImage image, SegmentationOptions segmentationOptions) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    inputPackets.put(
        OUTPUT_SIZE_IN_STREAM_NAME,
        runner
            .getPacketCreator()
            .createInt32Pair(
                segmentationOptions.outputWidth(), segmentationOptions.outputHeight()));
    if (!normRectStreamName.isEmpty()) {
      inputPackets.put(
          normRectStreamName,
          runner
              .getPacketCreator()
              .createProto(
                  convertToNormalizedRect(segmentationOptions.imageProcessingOptions(), image)));
    }
    return inputPackets;
  }
}
