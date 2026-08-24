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

package com.google.mediapipe.tasks.vision.imageembedder;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import androidx.annotation.Nullable;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Embedding;
import com.google.mediapipe.tasks.components.containers.proto.EmbeddingsProto;
import com.google.mediapipe.tasks.components.processors.proto.EmbedderOptionsProto;
import com.google.mediapipe.tasks.components.utils.CosineSimilarity;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler.ResultListener;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.imageembedder.proto.ImageEmbedderGraphOptionsProto;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

/**
 * Performs embedding extraction on images.
 *
 * <p>The API expects a TFLite model with optional, but strongly recommended, <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <p>The API supports models with one image input tensor and one or more output tensors. To be more
 * specific, here are the requirements.
 *
 * <ul>
 *   <li>Input image tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32})
 *       <ul>
 *         <li>image input of size {@code [batch x height x width x channels]}.
 *         <li>batch inference is not supported ({@code batch} is required to be 1).
 *         <li>only RGB inputs are supported ({@code channels} is required to be 3).
 *         <li>if type is kTfLiteFloat32, NormalizationOptions are required to be attached to the
 *             metadata for input normalization.
 *       </ul>
 *   <li>At least one output tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32}) with shape {@code
 *       [1 x N]} where N is the number of dimensions in the produced embeddings.
 * </ul>
 */
public final class ImageEmbedder implements AutoCloseable {
  private static final String TAG = ImageEmbedder.class.getSimpleName();
  private final ImageEmbedderExecutor executor;

  static {
    System.loadLibrary("mediapipe_tasks_jni");
    ProtoUtil.registerTypeName(
        EmbeddingsProto.EmbeddingResult.class,
        "mediapipe.tasks.components.containers.proto.EmbeddingResult");
  }

  /**
   * Creates an {@link ImageEmbedder} instance from a model file and default {@link
   * ImageEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelPath path to the embedding model in the assets.
   * @throws MediaPipeException if there is an error during {@link ImageEmbedder} creation.
   */
  public static ImageEmbedder createFromFile(Context context, String modelPath) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetPath(modelPath).build();
    return createFromOptions(
        context, ImageEmbedderOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ImageEmbedder} instance from a model file and default {@link
   * ImageEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelFile the embedding model {@link File} instance.
   * @throws IOException if an I/O error occurs when opening the tflite model file.
   * @throws MediaPipeException if there is an error during {@link ImageEmbedder} creation.
   */
  public static ImageEmbedder createFromFile(Context context, File modelFile) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      BaseOptions baseOptions =
          BaseOptions.builder().setModelAssetFileDescriptor(descriptor.getFd()).build();
      return createFromOptions(
          context, ImageEmbedderOptions.builder().setBaseOptions(baseOptions).build());
    }
  }

  /**
   * Creates an {@link ImageEmbedder} instance from a model buffer and default {@link
   * ImageEmbedderOptions}.
   *
   * @param context an Android {@link Context}.
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the embedding
   *     model.
   * @throws MediaPipeException if there is an error during {@link ImageEmbedder} creation.
   */
  public static ImageEmbedder createFromBuffer(Context context, final ByteBuffer modelBuffer) {
    BaseOptions baseOptions = BaseOptions.builder().setModelAssetBuffer(modelBuffer).build();
    return createFromOptions(
        context, ImageEmbedderOptions.builder().setBaseOptions(baseOptions).build());
  }

  /**
   * Creates an {@link ImageEmbedder} instance from an {@link ImageEmbedderOptions} instance.
   *
   * @param context an Android {@link Context}.
   * @param options an {@link ImageEmbedderOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageEmbedder} creation.
   */
  public static ImageEmbedder createFromOptions(Context context, ImageEmbedderOptions options) {
    return new ImageEmbedder(createGraphExecutor(context, options));
  }

  private static ImageEmbedderExecutor createGraphExecutor(
      Context context, ImageEmbedderOptions options) {
    return ImageEmbedderGraphExecutorImpl.create(context, options);
  }

  private ImageEmbedder(ImageEmbedderExecutor executor) {
    this.executor = executor;
  }

  /**
   * Performs embedding extraction on the provided single image with default image processing
   * options, i.e. using the whole image as region-of-interest and without any rotation applied.
   * Only use this method when the {@link ImageEmbedder} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageEmbedderResult embed(MPImage image) {
    return embed(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs embedding extraction on the provided single image. Only use this method when the
   * {@link ImageEmbedder} is created with {@link RunningMode.IMAGE}.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageEmbedderResult embed(MPImage image, ImageProcessingOptions imageProcessingOptions) {
    return executor.embed(image, imageProcessingOptions);
  }

  /**
   * Performs embedding extraction on the provided video frame with default image processing
   * options, i.e. using the whole image as region-of-interest and without any rotation applied.
   * Only use this method when the {@link ImageEmbedder} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageEmbedderResult embedForVideo(MPImage image, long timestampMs) {
    return embedForVideo(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Performs embedding extraction on the provided video frame. Only use this method when the {@link
   * ImageEmbedder} is created with {@link RunningMode.VIDEO}.
   *
   * <p>It's required to provide the video frame's timestamp (in milliseconds). The input timestamps
   * must be monotonically increasing.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public ImageEmbedderResult embedForVideo(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    return executor.embedForVideo(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Sends live image data to perform embedding extraction with default image processing options,
   * i.e. using the whole image as region-of-interest and without any rotation applied, and the
   * results will be available via the {@link ResultListener} provided in the {@link
   * ImageEmbedderOptions}. Only use this method when the {@link ImageEmbedder} is created with
   * {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void embedAsync(MPImage image, long timestampMs) {
    embedAsync(image, ImageProcessingOptions.builder().build(), timestampMs);
  }

  /**
   * Sends live image data to perform embedding extraction, and the results will be available via
   * the {@link ResultListener} provided in the {@link ImageEmbedderOptions}. Only use this method
   * when the {@link ImageEmbedder} is created with {@link RunningMode.LIVE_STREAM}.
   *
   * <p>It's required to provide a timestamp (in milliseconds) to indicate when the input image is
   * sent to the object detector. The input timestamps must be monotonically increasing.
   *
   * <p>{@link ImageEmbedder} supports the following color space types:
   *
   * <ul>
   *   <li>{@link Bitmap.Config.ARGB_8888}
   * </ul>
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the input timestamp (in milliseconds).
   * @throws MediaPipeException if there is an internal error.
   */
  public void embedAsync(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    executor.embedAsync(image, imageProcessingOptions, timestampMs);
  }

  /**
   * Utility function to compute <a href="https://en.wikipedia.org/wiki/Cosine_similarity">cosine
   * similarity</a> between two {@link Embedding} objects.
   *
   * @throws IllegalArgumentException if the embeddings are of different types (float vs.
   *     quantized), have different sizes, or have an L2-norm of 0.
   */
  public static double cosineSimilarity(Embedding u, Embedding v) {
    return CosineSimilarity.compute(u, v);
  }

  /** Options for setting up and {@link ImageEmbedder}. */
  @AutoValue
  public abstract static class ImageEmbedderOptions extends TaskOptions {

    /** Builder for {@link ImageEmbedderOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {
      /** Sets the {@link BaseOptions} for the image embedder task. */
      public abstract Builder setBaseOptions(BaseOptions baseOptions);

      /**
       * Sets the {@link RunningMode} for the image embedder task. Default to the image mode. Image
       * embedder has three modes:
       *
       * <ul>
       *   <li>IMAGE: The mode for performing embedding extraction on single image inputs.
       *   <li>VIDEO: The mode for performing embedding extraction on the decoded frames of a video.
       *   <li>LIVE_STREAM: The mode for for performing embedding extraction on a live stream of
       *       input data, such as from camera. In this mode, {@code setResultListener} must be
       *       called to set up a listener to receive the embedding results asynchronously.
       * </ul>
       */
      public abstract Builder setRunningMode(RunningMode runningMode);

      /**
       * Sets whether L2 normalization should be performed on the returned embeddings. Use this
       * option only if the model does not already contain a native <code>L2_NORMALIZATION</code> TF
       * Lite Op. In most cases, this is already the case and L2 norm is thus achieved through TF
       * Lite inference.
       *
       * <p>False by default.
       */
      public abstract Builder setL2Normalize(boolean l2Normalize);

      /**
       * Sets whether the returned embedding should be quantized to bytes via scalar quantization.
       * Embeddings are implicitly assumed to be unit-norm and therefore any dimensions is
       * guaranteed to have value in <code>[-1.0, 1.0]</code>. Use {@link #setL2Normalize(boolean)}
       * if this is not the case.
       *
       * <p>False by default.
       */
      public abstract Builder setQuantize(boolean quantize);

      /**
       * Sets the {@link ResultListener} to receive the embedding results asynchronously, when the
       * image embedder is in the live stream mode.
       */
      public abstract Builder setResultListener(
          ResultListener<ImageEmbedderResult, MPImage> resultListener);

      /** Sets an optional {@link ErrorListener}. */
      public abstract Builder setErrorListener(ErrorListener errorListener);

      abstract ImageEmbedderOptions autoBuild();

      /**
       * Validates and builds the {@link ImageEmbedderOptions} instance. *
       *
       * @throws IllegalArgumentException if the result listener and the running mode are not
       *     properly configured. The result listener should only be set when the image embedder is
       *     in the live stream mode.
       */
      public final ImageEmbedderOptions build() {
        ImageEmbedderOptions options = autoBuild();
        if (options.runningMode() == RunningMode.LIVE_STREAM) {
          if (!options.resultListener().isPresent()) {
            throw new IllegalArgumentException(
                "The image embedder is in the live stream mode, a user-defined result listener"
                    + " must be provided in the ImageEmbedderOptions.");
          }
        } else if (options.resultListener().isPresent()) {
          throw new IllegalArgumentException(
              "The image embedder is in the image or video mode, a user-defined result listener"
                  + " shouldn't be provided in ImageEmbedderOptions.");
        }
        return options;
      }
    }

    abstract BaseOptions baseOptions();

    abstract RunningMode runningMode();

    abstract boolean l2Normalize();

    abstract boolean quantize();

    abstract Optional<ResultListener<ImageEmbedderResult, MPImage>> resultListener();

    abstract Optional<ErrorListener> errorListener();

    /** Instantiates a builder for {@link ImageEmbedderOptions}. */
    public static Builder builder() {
      return new AutoValue_ImageEmbedder_ImageEmbedderOptions.Builder()
          .setRunningMode(RunningMode.IMAGE)
          .setL2Normalize(false)
          .setQuantize(false);
    }

    /** Converts a {@link ImageEmbedderOptions} to a {@link CalculatorOptions} protobuf message. */
    @Override
    public CalculatorOptions convertToCalculatorOptionsProto() {
      BaseOptionsProto.BaseOptions.Builder baseOptionsBuilder =
          BaseOptionsProto.BaseOptions.newBuilder();
      baseOptionsBuilder.setUseStreamMode(runningMode() != RunningMode.IMAGE);
      baseOptionsBuilder.mergeFrom(convertBaseOptionsToProto(baseOptions()));
      EmbedderOptionsProto.EmbedderOptions.Builder embedderOptionsBuilder =
          EmbedderOptionsProto.EmbedderOptions.newBuilder();
      embedderOptionsBuilder.setL2Normalize(l2Normalize());
      embedderOptionsBuilder.setQuantize(quantize());
      ImageEmbedderGraphOptionsProto.ImageEmbedderGraphOptions.Builder taskOptionsBuilder =
          ImageEmbedderGraphOptionsProto.ImageEmbedderGraphOptions.newBuilder()
              .setBaseOptions(baseOptionsBuilder)
              .setEmbedderOptions(embedderOptionsBuilder);
      return CalculatorOptions.newBuilder()
          .setExtension(
              ImageEmbedderGraphOptionsProto.ImageEmbedderGraphOptions.ext,
              taskOptionsBuilder.build())
          .build();
    }
  }
}
