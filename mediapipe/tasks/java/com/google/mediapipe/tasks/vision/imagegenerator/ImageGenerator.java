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

package com.google.mediapipe.tasks.vision.imagegenerator;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;
import androidx.annotation.Nullable;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.calculator.proto.StableDiffusionIterateCalculatorOptionsProto;
import com.google.mediapipe.proto.CalculatorOptionsProto.CalculatorOptions;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.OutputHandler;
import com.google.mediapipe.tasks.core.TaskInfo;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.TaskResult;
import com.google.mediapipe.tasks.core.TaskRunner;
import com.google.mediapipe.tasks.core.proto.ExternalFileProto;
import com.google.mediapipe.tasks.vision.core.BaseVisionTaskApi;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker.FaceLandmarkerOptions;
import com.google.mediapipe.tasks.vision.facelandmarker.proto.FaceLandmarkerGraphOptionsProto.FaceLandmarkerGraphOptions;
import com.google.mediapipe.tasks.vision.imagegenerator.proto.ConditionedImageGraphOptionsProto.ConditionedImageGraphOptions;
import com.google.mediapipe.tasks.vision.imagegenerator.proto.ControlPluginGraphOptionsProto;
import com.google.mediapipe.tasks.vision.imagegenerator.proto.ImageGeneratorGraphOptionsProto;
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter.ImageSegmenterOptions;
import com.google.mediapipe.tasks.vision.imagesegmenter.proto.ImageSegmenterGraphOptionsProto.ImageSegmenterGraphOptions;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Performs image generation from a text prompt. */
public final class ImageGenerator extends BaseVisionTaskApi {

  private static final String STEPS_STREAM_NAME = "steps";
  private static final String ITERATION_STREAM_NAME = "iteration";
  private static final String PROMPT_STREAM_NAME = "prompt";
  private static final String RAND_SEED_STREAM_NAME = "rand_seed";
  private static final String SOURCE_CONDITION_IMAGE_STREAM_NAME = "source_condition_image";
  private static final String CONDITION_IMAGE_STREAM_NAME = "condition_image";
  private static final String SELECT_STREAM_NAME = "select";
  private static final String SHOW_RESULT_STREAM_NAME = "show_result";
  private static final int GENERATED_IMAGE_OUT_STREAM_INDEX = 0;
  private static final int STEPS_OUT_STREAM_INDEX = 1;
  private static final int ITERATION_OUT_STREAM_INDEX = 2;
  private static final int SHOW_RESULT_OUT_STREAM_INDEX = 3;
  private static final String TASK_GRAPH_NAME =
      "mediapipe.tasks.vision.image_generator.ImageGeneratorGraph";
  private static final String CONDITION_IMAGE_GRAPHS_CONTAINER_NAME =
      "mediapipe.tasks.vision.image_generator.ConditionedImageGraphContainer";
  private static final String TAG = "ImageGenerator";
  private static final int GENERATED_IMAGE_WIDTH = 512;
  private static final int GENERATED_IMAGE_HEIGHT = 512;
  private TaskRunner conditionImageGraphsContainerTaskRunner;
  private Map<ConditionOptions.ConditionType, Integer> conditionTypeIndex;
  private boolean useConditionImage = false;
  private CachedInputs cachedInputs = new CachedInputs();
  private boolean inProcessing = false;

  static {
    System.loadLibrary("mediapipe_tasks_vision_image_generator_jni");
  }

  /**
   * Creates an {@link ImageGenerator} instance from an {@link ImageGeneratorOptions}.
   *
   * @param context an Android {@link Context}.
   * @param generatorOptions an {@link ImageGeneratorOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageGenerator} creation.
   */
  public static ImageGenerator createFromOptions(
      Context context, ImageGeneratorOptions generatorOptions) {
    return createFromOptions(context, generatorOptions, null);
  }

  /**
   * Creates an {@link ImageGenerator} instance, from {@link ImageGeneratorOptions} and {@link
   * ConditionOptions}, if plugin models are used to generate an image based on the condition image.
   *
   * @param context an Android {@link Context}.
   * @param generatorOptions an {@link ImageGeneratorOptions} instance.
   * @param conditionOptions an {@link ConditionOptions} instance.
   * @throws MediaPipeException if there is an error during {@link ImageGenerator} creation.
   */
  public static ImageGenerator createFromOptions(
      Context context,
      ImageGeneratorOptions generatorOptions,
      @Nullable ConditionOptions conditionOptions) {
    List<String> inputStreams = new ArrayList<>();
    inputStreams.addAll(
        Arrays.asList(
            "STEPS:" + STEPS_STREAM_NAME,
            "ITERATION:" + ITERATION_STREAM_NAME,
            "PROMPT:" + PROMPT_STREAM_NAME,
            "RAND_SEED:" + RAND_SEED_STREAM_NAME,
            "SHOW_RESULT:" + SHOW_RESULT_STREAM_NAME));
    final boolean useConditionImage = conditionOptions != null;
    if (useConditionImage) {
      inputStreams.add("SELECT:" + SELECT_STREAM_NAME);
      inputStreams.add("CONDITION_IMAGE:" + CONDITION_IMAGE_STREAM_NAME);
      generatorOptions.conditionOptions = Optional.of(conditionOptions);
    }
    List<String> outputStreams =
        Arrays.asList(
            "IMAGE:image_out",
            "STEPS:steps_out",
            "ITERATION:iteration_out",
            "SHOW_RESULT:show_result_out");

    OutputHandler<ImageGeneratorResult, Void> handler = new OutputHandler<>();
    handler.setOutputPacketConverter(
        new OutputHandler.OutputPacketConverter<ImageGeneratorResult, Void>() {
          @Override
          @Nullable
          public ImageGeneratorResult convertToTaskResult(List<Packet> packets) {
            int iteration = PacketGetter.getInt32(packets.get(ITERATION_OUT_STREAM_INDEX));
            int steps = PacketGetter.getInt32(packets.get(STEPS_OUT_STREAM_INDEX));
            boolean showResult =
                PacketGetter.getBool(packets.get(SHOW_RESULT_OUT_STREAM_INDEX))
                    || iteration == steps - 1;
            Log.i(
                "ImageGenerator",
                "Iteration: " + iteration + ", Steps: " + steps + ", ShowResult: " + showResult);
            if (showResult) {
              Log.i("ImageGenerator", "processing generated image");
              Packet packet = packets.get(GENERATED_IMAGE_OUT_STREAM_INDEX);
              Bitmap generatedBitmap = AndroidPacketGetter.getBitmapFromRgb(packet);
              BitmapImageBuilder bitmapImageBuilder = new BitmapImageBuilder(generatedBitmap);
              return ImageGeneratorResult.create(
                  bitmapImageBuilder.build(), packet.getTimestamp() / MICROSECONDS_PER_MILLISECOND);
            } else {
              return null;
            }
          }

          @Override
          public Void convertToTaskInput(List<Packet> packets) {
            return null;
          }
        });
    handler.setHandleTimestampBoundChanges(true);
    generatorOptions.errorListener().ifPresent(handler::setErrorListener);
    TaskRunner runner =
        TaskRunner.create(
            context,
            TaskInfo.<ImageGeneratorOptions>builder()
                .setTaskName(ImageGenerator.class.getSimpleName())
                .setTaskRunningModeName(RunningMode.IMAGE.name())
                .setTaskGraphName(TASK_GRAPH_NAME)
                .setInputStreams(inputStreams)
                .setOutputStreams(outputStreams)
                .setTaskOptions(generatorOptions)
                .setEnableFlowLimiting(false)
                .build(),
            handler);
    ImageGenerator imageGenerator = new ImageGenerator(runner);
    if (useConditionImage) {
      imageGenerator.useConditionImage = true;
      inputStreams =
          Arrays.asList(
              "IMAGE:" + SOURCE_CONDITION_IMAGE_STREAM_NAME, "SELECT:" + SELECT_STREAM_NAME);
      outputStreams = Arrays.asList("CONDITION_IMAGE:" + CONDITION_IMAGE_STREAM_NAME);
      OutputHandler<ConditionImageResult, Void> conditionImageHandler = new OutputHandler<>();
      conditionImageHandler.setOutputPacketConverter(
          new OutputHandler.OutputPacketConverter<ConditionImageResult, Void>() {
            @Override
            public ConditionImageResult convertToTaskResult(List<Packet> packets) {
              Packet packet = packets.get(0);
              return new AutoValue_ImageGenerator_ConditionImageResult(
                  new BitmapImageBuilder(AndroidPacketGetter.getBitmapFromRgb(packet)).build(),
                  packet.getTimestamp() / MICROSECONDS_PER_MILLISECOND);
            }

            @Override
            public Void convertToTaskInput(List<Packet> packets) {
              return null;
            }
          });
      conditionImageHandler.setHandleTimestampBoundChanges(true);
      imageGenerator.conditionImageGraphsContainerTaskRunner =
          TaskRunner.create(
              context,
              TaskInfo.<ImageGeneratorOptions>builder()
                  .setTaskName(ImageGenerator.class.getSimpleName())
                  .setTaskRunningModeName(RunningMode.IMAGE.name())
                  .setTaskGraphName(CONDITION_IMAGE_GRAPHS_CONTAINER_NAME)
                  .setInputStreams(inputStreams)
                  .setOutputStreams(outputStreams)
                  .setTaskOptions(generatorOptions)
                  .setEnableFlowLimiting(false)
                  .build(),
              conditionImageHandler);
      imageGenerator.conditionTypeIndex = new HashMap<>();
      if (conditionOptions.faceConditionOptions().isPresent()) {
        imageGenerator.conditionTypeIndex.put(
            ConditionOptions.ConditionType.FACE, imageGenerator.conditionTypeIndex.size());
      }
      if (conditionOptions.edgeConditionOptions().isPresent()) {
        imageGenerator.conditionTypeIndex.put(
            ConditionOptions.ConditionType.EDGE, imageGenerator.conditionTypeIndex.size());
      }
      if (conditionOptions.depthConditionOptions().isPresent()) {
        imageGenerator.conditionTypeIndex.put(
            ConditionOptions.ConditionType.DEPTH, imageGenerator.conditionTypeIndex.size());
      }
    }
    return imageGenerator;
  }

  private ImageGenerator(TaskRunner taskRunner) {
    super(taskRunner, RunningMode.IMAGE, "", "");
  }

  /**
   * Generates an image for iterations and the given random seed. Only valid when the ImageGenerator
   * is created without condition options.
   *
   * <p>This is an e2e API, which runs {@code iterations} to generate an image. Consider using the
   * iterative API instead to fetch the intermediate results.
   *
   * @param prompt The text prompt describing the image to be generated.
   * @param iterations The total iterations to generate the image.
   * @param seed The random seed used during image generation.
   */
  public ImageGeneratorResult generate(String prompt, int iterations, int seed) {
    if (useConditionImage) {
      throw new IllegalArgumentException(
          "ImageGenerator is created with condition options. Must use the methods with condition "
              + "options.");
    }
    return runIterations(prompt, iterations, seed, null, 0);
  }

  /**
   * Generates an image based on the source image for iterations and the given random seed. Only
   * valid when the ImageGenerator is created with condition options.
   *
   * <p>This is an e2e API, which runs {@code iterations} to generate an image. Consider using the
   * iterative API instead to fetch the intermediate results.
   *
   * @param prompt The text prompt describing the image to be generated.
   * @param sourceConditionImage The source image used to create the condition image, which is used
   *     as a guidance for the image generation.
   * @param conditionType The {@link ConditionOptions.ConditionType} specifying the type of
   *     condition image.
   * @param iterations The total iterations to generate the image.
   * @param seed The random seed used during image generation.
   */
  public ImageGeneratorResult generate(
      String prompt,
      MPImage sourceConditionImage,
      ConditionOptions.ConditionType conditionType,
      int iterations,
      int seed) {
    if (!useConditionImage) {
      throw new IllegalArgumentException(
          "ImageGenerator is created without condition options. Must use the methods without "
              + "condition options.");
    }
    return runIterations(
        prompt,
        iterations,
        seed,
        createConditionImage(sourceConditionImage, conditionType),
        conditionTypeIndex.get(conditionType));
  }

  /**
   * Sets the inputs of the ImageGenerator. There is {@link setInputs} and {@link execute} method
   * pair for iterative usage. Users must call {@link setInputs} before {@link execute}. Only valid
   * when the ImageGenerator is created without condition options.
   *
   * @param prompt The text prompt describing the image to be generated.
   * @param iterations The total iterations to generate the image.
   * @param seed The random seed used during image generation.
   */
  public void setInputs(String prompt, int iterations, int seed) {
    if (useConditionImage) {
      throw new IllegalArgumentException(
          "ImageGenerator is created with condition options. Must use the methods with condition "
              + "options.");
    }
    cachedInputs = new CachedInputs();
    cachedInputs.prompt = prompt;
    cachedInputs.iterations = iterations;
    cachedInputs.seed = seed;
    cachedInputs.step = 0;
    inProcessing = true;
  }

  /**
   * Sets the inputs of the ImageGenerator. For iterative usage, use {@link setInputs} and {@link
   * execute} in pairs. Users must call {@link setInputs} before {@link execute}. Only valid when
   * the ImageGenerator is created with condition options.
   *
   * @param prompt The text prompt describing the image to be generated.
   * @param sourceConditionImage The source image used to create the condition image, which is used
   *     as a guidance for the image generation.
   * @param conditionType The {@link ConditionOptions.ConditionType} specifying the type of
   *     condition image.
   * @param iterations The total iterations to generate the image.
   * @param seed The random seed used during image generation.
   */
  public void setInputs(
      String prompt,
      MPImage sourceConditionImage,
      ConditionOptions.ConditionType conditionType,
      int iterations,
      int seed) {
    if (!useConditionImage) {
      throw new IllegalArgumentException(
          "ImageGenerator is created without condition options. Must use the methods without "
              + "condition options.");
    }
    cachedInputs = new CachedInputs();
    cachedInputs.prompt = prompt;
    cachedInputs.iterations = iterations;
    cachedInputs.seed = seed;
    cachedInputs.step = 0;
    cachedInputs.cachedConditionImage = createConditionImage(sourceConditionImage, conditionType);
    cachedInputs.conditionType = conditionType;
    inProcessing = true;
  }

  /**
   * Executes one iteration of image generation. The method must be called {@code iterations} times
   * to generate the final image. Must call {@link setInputs} before calling this method.
   *
   * <p>This is an iterative API, which must be called iteratively.
   *
   * <p>This API is useful for showing the intermediate image generation results and the image
   * generation progress. Note that requesting the intermediate results will result in a larger
   * latency. Consider using the e2e API instead for latency consideration.
   *
   * <p>Example usage:
   *
   * <p>imageGenerator.setInputs(prompt, iterations, seed); for (int step = 0; step < iterations;
   * step++) { ImageGeneratorResult result = imageGenerator.execute(true); }
   *
   * @param showResult Whether to get the generated image result in the intermediate iterations. If
   *     false, null is returned. The generated image result is always returned at the last
   *     iteration, regardless of showResult value.
   */
  @Nullable
  public ImageGeneratorResult execute(boolean showResult) {
    if (!inProcessing) {
      throw new IllegalArgumentException("Must call setInputs before execute.");
    }
    return runStep(
        cachedInputs.prompt,
        cachedInputs.iterations,
        cachedInputs.step++,
        cachedInputs.seed,
        cachedInputs.cachedConditionImage,
        useConditionImage ? conditionTypeIndex.get(cachedInputs.conditionType) : 0,
        showResult);
  }

  /**
   * Create the condition image of specified condition type from the source image. Currently support
   * face landmarks, depth image and edge image as the condition image.
   *
   * @param sourceConditionImage The source image used to create the condition image.
   * @param conditionType The {@link ConditionOptions.ConditionType} specifying the type of
   *     condition image.
   */
  public MPImage createConditionImage(
      MPImage sourceConditionImage, ConditionOptions.ConditionType conditionType) {
    if (!conditionTypeIndex.containsKey(conditionType)) {
      throw new IllegalArgumentException(
          "The condition type " + conditionType.name() + " is not created during initialization.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(
        SOURCE_CONDITION_IMAGE_STREAM_NAME,
        conditionImageGraphsContainerTaskRunner
            .getPacketCreator()
            .createImage(sourceConditionImage));
    inputPackets.put(
        SELECT_STREAM_NAME,
        conditionImageGraphsContainerTaskRunner
            .getPacketCreator()
            .createInt32(conditionTypeIndex.get(conditionType)));
    ConditionImageResult result =
        (ConditionImageResult) conditionImageGraphsContainerTaskRunner.process(inputPackets);
    return result.conditionImage();
  }

  private ImageGeneratorResult runIterations(
      String prompt, int steps, int seed, @Nullable MPImage conditionImage, int select) {
    if (inProcessing) {
      throw new IllegalArgumentException(
          "Iterative API was called previously. It is not allowed to called batch API during"
              + "iterative processing.");
    }
    ImageGeneratorResult result = null;
    long timestamp = System.currentTimeMillis() * MICROSECONDS_PER_MILLISECOND;
    for (int i = 0; i < steps; i++) {
      Map<String, Packet> inputPackets = new HashMap<>();
      if (i == 0 && useConditionImage) {
        inputPackets.put(
            CONDITION_IMAGE_STREAM_NAME, runner.getPacketCreator().createImage(conditionImage));
        inputPackets.put(SELECT_STREAM_NAME, runner.getPacketCreator().createInt32(select));
      }
      inputPackets.put(PROMPT_STREAM_NAME, runner.getPacketCreator().createString(prompt));
      inputPackets.put(STEPS_STREAM_NAME, runner.getPacketCreator().createInt32(steps));
      inputPackets.put(ITERATION_STREAM_NAME, runner.getPacketCreator().createInt32(i));
      inputPackets.put(RAND_SEED_STREAM_NAME, runner.getPacketCreator().createInt32(seed));
      inputPackets.put(SHOW_RESULT_STREAM_NAME, runner.getPacketCreator().createBool(false));
      result = (ImageGeneratorResult) runner.process(inputPackets, timestamp++);
    }
    if (useConditionImage) {
      // Add condition image to the ImageGeneratorResult.
      return ImageGeneratorResult.create(
          result.generatedImage(), conditionImage, result.timestampMs());
    }
    return result;
  }

  @Nullable
  private ImageGeneratorResult runStep(
      String prompt,
      int iterations,
      int step,
      int seed,
      @Nullable MPImage conditionImage,
      int select,
      boolean showResult) {
    if (step == 0) {
      cachedInputs.cachedTimestamp = System.currentTimeMillis() * MICROSECONDS_PER_MILLISECOND;
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    if (step == 0 && useConditionImage) {
      inputPackets.put(
          CONDITION_IMAGE_STREAM_NAME, runner.getPacketCreator().createImage(conditionImage));
      inputPackets.put(SELECT_STREAM_NAME, runner.getPacketCreator().createInt32(select));
    }
    inputPackets.put(PROMPT_STREAM_NAME, runner.getPacketCreator().createString(prompt));
    inputPackets.put(STEPS_STREAM_NAME, runner.getPacketCreator().createInt32(iterations));
    inputPackets.put(ITERATION_STREAM_NAME, runner.getPacketCreator().createInt32(step));
    inputPackets.put(RAND_SEED_STREAM_NAME, runner.getPacketCreator().createInt32(seed));
    inputPackets.put(SHOW_RESULT_STREAM_NAME, runner.getPacketCreator().createBool(showResult));
    ImageGeneratorResult result =
        (ImageGeneratorResult) runner.process(inputPackets, cachedInputs.cachedTimestamp++);
    if (result != null && useConditionImage) {
      // Add condition image to the ImageGeneratorResult.
      result =
          ImageGeneratorResult.create(
              result.generatedImage(), conditionImage, result.timestampMs());
    }
    if (step == iterations - 1) {
      inProcessing = false;
      cachedInputs = new CachedInputs();
    }
    return result;
  }

  /** Closes and cleans up the task runners. */
  @Override
  public void close() {
    if (runner != null) {
      runner.close();
    }
    if (conditionImageGraphsContainerTaskRunner != null) {
      conditionImageGraphsContainerTaskRunner.close();
    }
  }

  // Helper class to holder inputs to be checked with the inputs of next step.
  private static class CachedInputs {
    public CachedInputs() {
      this.prompt = "";
      this.iterations = 0;
      this.step = 0;
      this.seed = 0;
    }

    @Override
    public final String toString() {
      return "Prompt: "
          + prompt
          + ", Iterations: "
          + iterations
          + ", Step: "
          + step
          + ", Seed: "
          + seed
          + (conditionType == null ? "" : ", ConditionType: " + conditionType.name());
    }

    public String prompt;
    public int iterations;
    public int step;
    public int seed;
    public ConditionOptions.ConditionType conditionType;
    public MPImage cachedConditionImage;
    public long cachedTimestamp;
  }

  /** A container class for the condition image. */
  @AutoValue
  protected abstract static class ConditionImageResult implements TaskResult {

    public abstract MPImage conditionImage();

    @Override
    public abstract long timestampMs();
  }

  /** Options for setting up an {@link ImageGenerator}. */
  @AutoValue
  public abstract static class ImageGeneratorOptions extends TaskOptions {

    /** The supported model types. */
    public enum ModelType {
      SD_1, // Stable Diffusion v1 models, including SD 1.4 and 1.5.
    }

    /** Builder for {@link ImageGeneratorOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /** Sets the image generator model directory storing the model weights. */
      public abstract Builder setImageGeneratorModelDirectory(String modelDirectory);

      /** Sets the path to LoRA weights file. */
      public abstract Builder setLoraWeightsFilePath(String loraWeightsFilePath);

      /** Sets the model type. */
      public abstract Builder setModelType(ModelType modelType);

      /** Sets an optional {@link ErrorListener}}. */
      public abstract Builder setErrorListener(ErrorListener value);

      abstract ImageGeneratorOptions autoBuild();

      /** Validates and builds the {@link ImageGeneratorOptions} instance. */
      public final ImageGeneratorOptions build() {
        return autoBuild();
      }
    }

    abstract String imageGeneratorModelDirectory();

    abstract Optional<String> loraWeightsFilePath();

    abstract ModelType modelType();

    abstract Optional<ErrorListener> errorListener();

    private Optional<ConditionOptions> conditionOptions;

    private StableDiffusionIterateCalculatorOptionsProto.StableDiffusionIterateCalculatorOptions
            .ModelType
        convertModelTypeToProto(ModelType modelType) {
      switch (modelType) {
        case SD_1:
          return StableDiffusionIterateCalculatorOptionsProto
              .StableDiffusionIterateCalculatorOptions.ModelType.SD_1;
      }
      throw new IllegalArgumentException("Unsupported model type: " + modelType.name());
    }

    public static Builder builder() {
      return new AutoValue_ImageGenerator_ImageGeneratorOptions.Builder()
          .setImageGeneratorModelDirectory("")
          .setModelType(ModelType.SD_1);
    }

    /** Converts an {@link ImageGeneratorOptions} to a {@link Any} protobuf message. */
    @Override
    @SuppressWarnings("ProtoParseWithRegistry")
    public Any convertToAnyProto() {
      ImageGeneratorGraphOptionsProto.ImageGeneratorGraphOptions.Builder taskOptionsBuilder =
          ImageGeneratorGraphOptionsProto.ImageGeneratorGraphOptions.newBuilder();
      if (conditionOptions != null && conditionOptions.isPresent()) {
        try {
          taskOptionsBuilder.mergeFrom(conditionOptions.get().convertToAnyProto().getValue());
        } catch (InvalidProtocolBufferException e) {
          Log.e(TAG, "Error converting ConditionOptions to proto. " + e.getMessage());
          e.printStackTrace();
        }
      }
      taskOptionsBuilder.setText2ImageModelDirectory(imageGeneratorModelDirectory());
      if (loraWeightsFilePath().isPresent()) {
        ExternalFileProto.ExternalFile.Builder externalFileBuilder =
            ExternalFileProto.ExternalFile.newBuilder();
        externalFileBuilder.setFileName(loraWeightsFilePath().get());
        taskOptionsBuilder.setLoraWeightsFile(externalFileBuilder.build());
      }
      taskOptionsBuilder.setStableDiffusionIterateOptions(
          StableDiffusionIterateCalculatorOptionsProto.StableDiffusionIterateCalculatorOptions
              .newBuilder()
              .setBaseSeed(0)
              .setFileFolder(imageGeneratorModelDirectory())
              .setModelType(convertModelTypeToProto(modelType()))
              .setOutputImageWidth(GENERATED_IMAGE_WIDTH)
              .setOutputImageHeight(GENERATED_IMAGE_HEIGHT)
              .setEmitEmptyPacket(true)
              .setShowEveryNIteration(100)
              .build());
      return Any.newBuilder()
          .setTypeUrl(
              "type.googleapis.com/mediapipe.tasks.vision.image_generator.proto.ImageGeneratorGraphOptions")
          .setValue(taskOptionsBuilder.build().toByteString())
          .build();
    }
  }

  /** Options for setting up the conditions types and the plugin models */
  @AutoValue
  public abstract static class ConditionOptions extends TaskOptions {

    /** The supported condition type. */
    public enum ConditionType {
      FACE,
      EDGE,
      DEPTH
    }

    /** Builder for {@link ConditionOptions}. At least one type of condition options must be set. */
    @AutoValue.Builder
    public abstract static class Builder {
      public abstract Builder setFaceConditionOptions(FaceConditionOptions faceConditionOptions);

      public abstract Builder setDepthConditionOptions(DepthConditionOptions depthConditionOptions);

      public abstract Builder setEdgeConditionOptions(EdgeConditionOptions edgeConditionOptions);

      abstract ConditionOptions autoBuild();

      /** Validates and builds the {@link ConditionOptions} instance. */
      public final ConditionOptions build() {
        ConditionOptions options = autoBuild();
        if (!options.faceConditionOptions().isPresent()
            && !options.depthConditionOptions().isPresent()
            && !options.edgeConditionOptions().isPresent()) {
          throw new IllegalArgumentException(
              "At least one of `faceConditionOptions`, `depthConditionOptions` and"
                  + " `edgeConditionOptions` must be set.");
        }
        return options;
      }
    }

    abstract Optional<FaceConditionOptions> faceConditionOptions();

    abstract Optional<DepthConditionOptions> depthConditionOptions();

    abstract Optional<EdgeConditionOptions> edgeConditionOptions();

    public static Builder builder() {
      return new AutoValue_ImageGenerator_ConditionOptions.Builder();
    }

    /**
     * Converts an {@link ImageGeneratorOptions} to a {@link CalculatorOptions} protobuf message.
     */
    @Override
    public Any convertToAnyProto() {
      ImageGeneratorGraphOptionsProto.ImageGeneratorGraphOptions.Builder taskOptionsBuilder =
          ImageGeneratorGraphOptionsProto.ImageGeneratorGraphOptions.newBuilder();
      if (faceConditionOptions().isPresent()) {
        taskOptionsBuilder.addControlPluginGraphsOptions(
            ControlPluginGraphOptionsProto.ControlPluginGraphOptions.newBuilder()
                .setBaseOptions(
                    convertBaseOptionsToProto(
                        faceConditionOptions().get().pluginModelBaseOptions()))
                .setConditionedImageGraphOptions(
                    ConditionedImageGraphOptions.newBuilder()
                        .setFaceConditionTypeOptions(faceConditionOptions().get().convertToProto())
                        .build())
                .build());
      }
      if (edgeConditionOptions().isPresent()) {
        taskOptionsBuilder.addControlPluginGraphsOptions(
            ControlPluginGraphOptionsProto.ControlPluginGraphOptions.newBuilder()
                .setBaseOptions(
                    convertBaseOptionsToProto(
                        edgeConditionOptions().get().pluginModelBaseOptions()))
                .setConditionedImageGraphOptions(
                    ConditionedImageGraphOptions.newBuilder()
                        .setEdgeConditionTypeOptions(edgeConditionOptions().get().convertToProto())
                        .build())
                .build());
      }
      if (depthConditionOptions().isPresent()) {
        taskOptionsBuilder.addControlPluginGraphsOptions(
            ControlPluginGraphOptionsProto.ControlPluginGraphOptions.newBuilder()
                .setBaseOptions(
                    convertBaseOptionsToProto(
                        depthConditionOptions().get().pluginModelBaseOptions()))
                .setConditionedImageGraphOptions(
                    ConditionedImageGraphOptions.newBuilder()
                        .setDepthConditionTypeOptions(
                            depthConditionOptions().get().convertToProto())
                        .build())
                .build());
      }
      return Any.newBuilder()
          .setTypeUrl(
              "type.googleapis.com/mediapipe.tasks.vision.image_generator.proto.ImageGeneratorGraphOptions")
          .setValue(taskOptionsBuilder.build().toByteString())
          .build();
    }

    /** Options for drawing face landmarks image. */
    @AutoValue
    public abstract static class FaceConditionOptions extends TaskOptions {

      /** Builder for {@link FaceConditionOptions}. */
      @AutoValue.Builder
      public abstract static class Builder {
        /** Set the base options for plugin model. */
        public abstract Builder setPluginModelBaseOptions(BaseOptions baseOptions);

        /** Set base options for face landmarks model. */
        public abstract Builder setFaceModelBaseOptions(BaseOptions baseOptions);

        /** Set minimum confidence score of face presence score in the face landmark detection. */
        public abstract Builder setMinFaceDetectionConfidence(float minFaceDetectionConfidence);

        /** Set the face presence threshold */
        public abstract Builder setMinFacePresenceConfidence(float minFacePresenceConfidence);

        abstract FaceConditionOptions autoBuild();

        /** Validates and builds the {@link FaceConditionOptions} instance. */
        public final FaceConditionOptions build() {
          return autoBuild();
        }
      }

      abstract BaseOptions pluginModelBaseOptions();

      abstract BaseOptions faceModelBaseOptions();

      abstract float minFaceDetectionConfidence();

      abstract float minFacePresenceConfidence();

      public static Builder builder() {
        return new AutoValue_ImageGenerator_ConditionOptions_FaceConditionOptions.Builder()
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f);
      }

      ConditionedImageGraphOptions.FaceConditionTypeOptions convertToProto() {
        FaceLandmarkerOptions faceLandmarkerOptions =
            FaceLandmarkerOptions.builder()
                .setBaseOptions(faceModelBaseOptions())
                .setMinFaceDetectionConfidence(minFaceDetectionConfidence())
                .setMinFacePresenceConfidence(minFacePresenceConfidence())
                .setRunningMode(RunningMode.IMAGE)
                .setOutputFaceBlendshapes(false)
                .setOutputFacialTransformationMatrixes(false)
                .setNumFaces(1)
                .build();
        return ConditionedImageGraphOptions.FaceConditionTypeOptions.newBuilder()
            .setFaceLandmarkerGraphOptions(
                faceLandmarkerOptions
                    .convertToCalculatorOptionsProto()
                    .getExtension(FaceLandmarkerGraphOptions.ext))
            .build();
      }
    }

    /** Options for detecting depth image. */
    @AutoValue
    public abstract static class DepthConditionOptions extends TaskOptions {

      /** Builder for {@link DepthConditionOptions}. */
      @AutoValue.Builder
      public abstract static class Builder {

        /** Set the base options for the plugin model. */
        public abstract Builder setPluginModelBaseOptions(BaseOptions baseOptions);

        /** Set the base options for the depth model. */
        public abstract Builder setDepthModelBaseOptions(BaseOptions baseOptions);

        abstract DepthConditionOptions autoBuild();

        /** Validates and builds the {@link DepthConditionOptions} instance. */
        public final DepthConditionOptions build() {
          DepthConditionOptions options = autoBuild();
          return options;
        }
      }

      abstract BaseOptions pluginModelBaseOptions();

      abstract BaseOptions depthModelBaseOptions();

      public static Builder builder() {
        return new AutoValue_ImageGenerator_ConditionOptions_DepthConditionOptions.Builder();
      }

      ConditionedImageGraphOptions.DepthConditionTypeOptions convertToProto() {
        ImageSegmenterOptions imageSegmenterOptions =
            ImageSegmenterOptions.builder()
                .setBaseOptions(depthModelBaseOptions())
                .setOutputConfidenceMasks(true)
                .setOutputCategoryMask(false)
                .setRunningMode(RunningMode.IMAGE)
                .build();
        return ConditionedImageGraphOptions.DepthConditionTypeOptions.newBuilder()
            .setImageSegmenterGraphOptions(
                imageSegmenterOptions
                    .convertToCalculatorOptionsProto()
                    .getExtension(ImageSegmenterGraphOptions.ext))
            .build();
      }
    }

    /** Options for detecting edge image. */
    @AutoValue
    public abstract static class EdgeConditionOptions {

      /**
       * Builder for {@link EdgeConditionOptions}.
       *
       * <p>These parameters are used to config Canny edge algorithm of OpenCV.
       *
       * <p>See more details:
       * https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
       */
      @AutoValue.Builder
      public abstract static class Builder {

        /** Set the base options for plugin model. */
        public abstract Builder setPluginModelBaseOptions(BaseOptions baseOptions);

        /** First threshold for the hysteresis procedure. */
        public abstract Builder setThreshold1(Float threshold1);

        /** Second threshold for the hysteresis procedure. */
        public abstract Builder setThreshold2(Float threshold2);

        /** Aperture size for the Sobel operator. Typical range is 3~7. */
        public abstract Builder setApertureSize(Integer apertureSize);

        /**
         * flag, indicating whether a more accurate L2 norm should be used to calculate the image
         * gradient magnitude ( L2gradient=true ), or whether the default L1 norm is enough (
         * L2gradient=false ).
         */
        public abstract Builder setL2Gradient(Boolean l2Gradient);

        abstract EdgeConditionOptions autoBuild();

        /** Validates and builds the {@link EdgeConditionOptions} instance. */
        public final EdgeConditionOptions build() {
          return autoBuild();
        }
      }

      abstract BaseOptions pluginModelBaseOptions();

      abstract Float threshold1();

      abstract Float threshold2();

      abstract Integer apertureSize();

      abstract Boolean l2Gradient();

      public static Builder builder() {
        return new AutoValue_ImageGenerator_ConditionOptions_EdgeConditionOptions.Builder()
            .setThreshold1(100f)
            .setThreshold2(200f)
            .setApertureSize(3)
            .setL2Gradient(false);
      }

      ConditionedImageGraphOptions.EdgeConditionTypeOptions convertToProto() {
        return ConditionedImageGraphOptions.EdgeConditionTypeOptions.newBuilder()
            .setThreshold1(threshold1())
            .setThreshold2(threshold2())
            .setApertureSize(apertureSize())
            .setL2Gradient(l2Gradient())
            .build();
      }
    }
  }
}
