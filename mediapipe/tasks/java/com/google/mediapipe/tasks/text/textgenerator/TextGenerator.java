package com.google.mediapipe.tasks.text.textgenerator;

import android.content.Context;
import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.core.ErrorListener;
import com.google.mediapipe.tasks.core.LlmTaskRunner;
import com.google.mediapipe.tasks.core.OutputHandler.ValueListener;
import com.google.mediapipe.tasks.core.TaskOptions;
import com.google.mediapipe.tasks.core.jni.LlmOptionsProto.LlmModelParameters;
import com.google.mediapipe.tasks.core.jni.LlmOptionsProto.LlmModelParameters.LlmAttentionType;
import com.google.mediapipe.tasks.core.jni.LlmOptionsProto.LlmModelParameters.LlmModelType;
import com.google.mediapipe.tasks.core.jni.LlmOptionsProto.LlmSessionConfig;
import com.google.mediapipe.tasks.core.jni.LlmOptionsProto.LlmSessionConfig.LlmBackend;
import java.util.List;
import java.util.Optional;

/** TextGenerator Task Java API */
public final class TextGenerator implements AutoCloseable {
  private static final char TOKEN_SPLITTER = '▁'; // Note this is NOT an underscore: ▁(U+2581)
  private static final String NEW_LINE = "<0x0A>";
  private static final String EOD = "\\[eod\\]";

  private final LlmTaskRunner taskRunner;

  static {
    System.loadLibrary("llm_inference_engine_jni");
  }

  /** Creates a TextGenerator */
  public static TextGenerator createFromOptions(Context context, TextGeneratorOptions options) {
    // Configure LLM model parameters.
    LlmModelParameters.Builder modelParameters = LlmModelParameters.newBuilder();
    modelParameters.setModelDirectory(options.modelDirectory());

    switch (options.modelType()) {
      case FALCON_1B:
        modelParameters.setModelType(LlmModelType.FALCON_1B);
        break;
      case GMINI_2B:
        modelParameters.setModelType(LlmModelType.GMINI_2B);
        break;
    }

    modelParameters.setAttentionType(
        options.useMultiHeadAttention() ? LlmAttentionType.MHA : LlmAttentionType.MQA);

    modelParameters.setStartTokenId(options.startTokenId());
    if (options.stopTokens().isPresent()) {
      modelParameters.addAllStopTokens(options.stopTokens().get());
    }

    // Configure LLM session config.
    LlmSessionConfig.Builder sessionConfig = LlmSessionConfig.newBuilder();

    switch (options.delegate()) {
      case GPU:
        sessionConfig.setBackend(LlmBackend.GPU);
        break;
      case CPU:
        sessionConfig.setBackend(LlmBackend.CPU);
        break;
      default:
        throw new IllegalArgumentException("Only CPU or GPU is supported.");
    }

    sessionConfig.setUseFakeWeights(options.useFakeWeights());
    sessionConfig.setNumDecodeTokens(options.numDecodeTokens());
    sessionConfig.setMaxSequenceLength(options.maxSequenceLength());

    Optional<ValueListener<List<String>>> llmResultListener;
    if (options.resultListener().isPresent()) {
      llmResultListener =
          Optional.of(
              new ValueListener<List<String>>() {
                @Override
                public void run(List<String> tokens) {
                  String response = decodeResponse(tokens);
                  options.resultListener().get().run(response);
                }
              });
    } else {
      llmResultListener = Optional.empty();
    }

    LlmTaskRunner taskRunner =
        new LlmTaskRunner(modelParameters.build(), sessionConfig.build(), llmResultListener);
    return new TextGenerator(taskRunner);
  }

  /** Constructor to initialize a {@link TextGenerator}. */
  private TextGenerator(LlmTaskRunner taskRunner) {
    this.taskRunner = taskRunner;
  }

  /**
   * Generates a response based on the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public String generateResponse(String inputText) {
    List<String> tokens = taskRunner.predictSync(inputText);
    return decodeResponse(tokens);
  }

  /**
   * Generates a response based on the input text.
   *
   * @param inputText a {@link String} for processing.
   */
  public void generateResponseAsync(String inputText) {
    taskRunner.predictAsync(inputText);
  }

  /** Decodes the response from the LLM engine and returns a human-readable string. */
  private static String decodeResponse(List<String> responses) {
    if (responses.isEmpty()) {
      // Technically, this is an error. We should always get at least one response.
      return "";
    }

    String response = responses.get(0); // We only use the first response
    response = response.replace(TOKEN_SPLITTER, ' '); // Note this is NOT an underscore: ▁(U+2581)
    response = response.replace(NEW_LINE, "\n"); // Replace <0x0A> token with newline
    response = response.stripLeading(); // Strip all leading spaces for the first output

    return response.split(EOD, -1)[0];
  }

  /** Closes and cleans up the {@link TextGenerator}. */
  @Override
  public void close() {
    taskRunner.close();
  }

  /** Options for setting up an {@link TextGenerator}. */
  @AutoValue
  public abstract static class TextGeneratorOptions extends TaskOptions {

    /** Models supported by MediaPipe's TextGenerator */
    public enum ModelType {
      /** Falcon with 1B parameters. */
      FALCON_1B,
      /** GMini with 2B parameters. */
      GMINI_2B;
    }

    /** Builder for {@link TextGeneratorOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /** Sets the model directory for the text generator task. */
      public abstract Builder setModelDirectory(String modelDirectory);

      /** Sets the delegate for the text generator task. */
      public abstract Builder setDelegate(Delegate delegate);

      /** Sets the result listener to invoke with the async API. */
      public abstract Builder setResultListener(ValueListener<String> listener);

      /** Sets the error listener to invoke with the async API. */
      public abstract Builder setErrorListener(ErrorListener listener);

      /**
       * Sets the start token id to be prepended to the input prompt. This should match the settings
       * of the model training.
       */
      public abstract Builder setStartTokenId(int startTokenId);

      /**
       * Sets whether the model uses multi-head attention. If set to be {@code false}, MQA
       * (multi-query attention) will be used.
       */
      public abstract Builder setUseMultiHeadAttention(boolean useMultiHeadAttention);

      /** Sets the supported model types. */
      public abstract Builder setModelType(ModelType modelType);

      /** Sets the stop token/word to indicate that a response is complete. */
      public abstract Builder setStopTokens(List<String> topTokens);

      /** Sets the number of decoding steps to run for each GPU-CPU sync. */
      public abstract Builder setNumDecodeTokens(int numDecodeTokens);

      /**
       * Configures the sequence length stands (i.e. the total number of tokens from input and
       * output).
       */
      public abstract Builder setMaxSequenceLength(int maxSequenceLength);

      /** Configures whether to use fake weights instead of loading from file. */
      public abstract Builder setUseFakeWeights(boolean useFakeWeights);

      abstract TextGeneratorOptions autoBuild();

      /** Validates and builds the {@link ImageGeneratorOptions} instance. */
      public final TextGeneratorOptions build() {
        return autoBuild();
      }
    }

    /** The model type to use. */
    abstract ModelType modelType();

    /** The directory that stores the model data. */
    abstract String modelDirectory();

    /** The delegate to use. */
    abstract Delegate delegate();

    /**
     * The start token id to be prepended to the input prompt. This should match the settings of the
     * model training.
     */
    abstract int startTokenId();

    /**
     * Whether to use MQA (multi-query attention). If set to be {@code false}, MQA (multi-query
     * attention) will be used.
     */
    abstract boolean useMultiHeadAttention();

    /**
     * A list of tokens that signal the decoding should stop, e.g. </s>. Note that this argument
     * only takes effect when num_decode_tokens != -1.
     */
    abstract Optional<List<String>> stopTokens();

    /**
     * The number of decoding steps to run for each GPU-CPU sync. 1 stands for full streaming mode
     * (i.e. the model outputs one token at a time). -1 stands for non-streaming mode (i.e. the
     * model decodes all the way to the end and output the result at once). Note that the more
     * frequent to perform GPU-CPU sync (i.e. closer to 1), the more latency we expect to introduce.
     */
    abstract int numDecodeTokens();

    /**
     * The total length of the kv-cache. In other words, this is the total number of input + output
     * tokens the model needs to handle.
     */
    abstract int maxSequenceLength();

    /** Whether to use fake weights instead of loading from file. */
    public abstract boolean useFakeWeights();

    /** The result listener to use for the {@link TextGenerator#generateAsync} API. */
    abstract Optional<ValueListener<String>> resultListener();

    /** The error listener to use for the {@link TextGenerator#generateAsync} API. */
    abstract Optional<ErrorListener> errorListener();

    /** Instantiates a new TextGeneratorOptions builder. */
    public static Builder builder() {
      return new AutoValue_TextGenerator_TextGeneratorOptions.Builder()
          .setDelegate(Delegate.GPU)
          .setStartTokenId(-1)
          .setUseMultiHeadAttention(true)
          .setNumDecodeTokens(-1)
          .setMaxSequenceLength(512)
          .setUseFakeWeights(false);
    }
  }
}
