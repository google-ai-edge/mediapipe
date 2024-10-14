package com.google.mediapipe.tasks.genai.llminference;

import com.google.auto.value.AutoValue;
import com.google.mediapipe.tasks.core.LlmTaskRunner;
import com.google.mediapipe.tasks.core.LlmTaskRunner.LlmSession;
import com.google.mediapipe.tasks.core.jni.proto.LlmOptionsProto.LlmSessionConfig;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

/**
 * LlmInferenceSession Task Java API.
 *
 * <p>This class encapsulates an LLM inference session along with its context. The context can be
 * shared efficiently across multiple calls to {@link #generateResponse()} or {@link
 * #generateResponseAsync()} and can also be shared across multiple sessions using {@link
 * #cloneSession()}.
 */
public final class LlmInferenceSession implements AutoCloseable {
  private static final char TOKEN_SPLITTER = '▁'; // Note this is NOT an underscore: ▁(U+2581)
  private static final String NEW_LINE = "<0x0A>";
  private static final String EOD = "\\[eod\\]";

  private final LlmTaskRunner taskRunner;
  private final LlmSession session;

  /** Constructor to initialize an {@link LlmInferenceSession}. */
  public static LlmInferenceSession createFromOptions(
      LlmInference llmInference, LlmInferenceSessionOptions options) {
    LlmSessionConfig.Builder sessionConfig = LlmSessionConfig.newBuilder();
    sessionConfig.setTopk(options.topK());
    sessionConfig.setTemperature(options.temperature());
    sessionConfig.setRandomSeed(options.randomSeed());
    if (options.loraPath().isPresent()) {
      sessionConfig.setLoraPath(options.loraPath().get());
    } else {
      sessionConfig.setLoraPath("");
    }

    LlmTaskRunner taskRunner = llmInference.getTaskRunner();
    LlmSession session = taskRunner.createSession(sessionConfig.build());
    return new LlmInferenceSession(taskRunner, session);
  }

  private LlmInferenceSession(LlmTaskRunner taskRunner, LlmSession session) {
    this.taskRunner = taskRunner;
    this.session = session;
  }

  /**
   * Adds a query chunk to the session. This method can be called multiple times to add multiple
   * query chunks before calling {@link #generateResponse()} or {@link #generateResponseAsync()} .
   * The query chunks will be processed in the order they are added, similar to a concatenated
   * prompt, but able to be processed in chunks.
   *
   * @param inputText The query chunk to add to the session.
   * @throws MediaPipeException if adding a query chunk to the session fails.
   */
  public void addQueryChunk(String inputText) {
    taskRunner.addQueryChunk(session, inputText);
  }

  /**
   * Generates a response based on the previously added query chunks synchronously. Use {@link
   * #addQueryChunk(String)} to add at least one query chunk before calling this function.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @throws MediaPipeException if the inference fails.
   */
  public String generateResponse() {
    List<String> tokens = Collections.unmodifiableList(taskRunner.predictSync(session));
    return decodeResponse(tokens, /* stripLeadingWhitespace= */ true);
  }

  /**
   * Generates a response based on the previously added query chunks asynchronously.
   *
   * <p>The {@code resultListener} callback of the {@link LlmInference} instance returns the partial
   * responses from the LLM. Use {@link #addQueryChunk(String)} to add at least one query chunk
   * before calling this function.
   *
   * <p>Note: You cannot invoke simultaneous response generation calls on active sessions created
   * using the same {@link LlmInference}. You have to wait for the currently running response
   * generation call to complete before initiating another one.
   *
   * @throws MediaPipeException if the inference fails.
   */
  public void generateResponseAsync() {
    taskRunner.predictAsync(session);
  }

  /**
   * Runs an invocation of <b>only</b> the tokenization for the LLM, and returns the size (in
   * tokens) of the result. Cannot be called while a other calls are in progress.
   *
   * @param text The text to tokenize.
   * @return The number of tokens in the resulting tokenization of the text.
   * @throws MediaPipeException if the tokenization fails.
   */
  public int sizeInTokens(String text) {
    return taskRunner.sizeInTokens(session, text);
  }

  /** Decodes the response from the LLM engine and returns a human-readable string. */
  static String decodeResponse(List<String> responses, boolean stripLeadingWhitespace) {
    if (responses.isEmpty()) {
      // Technically, this is an error. We should always get at least one response.
      return "";
    }

    String response = responses.get(0); // We only use the first response
    response = response.replace(TOKEN_SPLITTER, ' '); // Note this is NOT an underscore: ▁(U+2581)
    response = response.replace(NEW_LINE, "\n"); // Replace <0x0A> token with newline

    if (stripLeadingWhitespace) {
      response = stripLeading(response); // Strip all leading spaces for the first output
    }

    return response.split(EOD, -1)[0];
  }

  // TODO: b/368657051 - Make this model public once we can clone GPU sessions on Android.
  /**
   * Clones the current session.
   *
   * <p>You can continue prompting the LLM from where you left off using the cloned session.
   *
   * @return A new instance of `Session` which is cloned from the current session.
   * @throws MediaPipeException if cloning the current session fails.
   */
  LlmInferenceSession cloneSession() {
    LlmSession clonedSession = taskRunner.cloneSession(session);
    return new LlmInferenceSession(taskRunner, clonedSession);
  }

  /** Closes and cleans up the {@link LlmInferenceSession}. */
  @Override
  public void close() {
    taskRunner.deleteSession(session);
  }

  /** Options for setting up an {@link LlmInferenceSessionOptions}. */
  @AutoValue
  public abstract static class LlmInferenceSessionOptions {
    /** Builder for {@link LlmInferenceSessionOptions}. */
    @AutoValue.Builder
    public abstract static class Builder {

      /**
       * Configures the top K number of tokens to be sampled from for each decoding step. A value of
       * 1 means greedy decoding. The default value is 40.
       */
      public abstract Builder setTopK(int topK);

      /**
       * Configures randomness when decoding the next token. A value of 0.0f means greedy decoding.
       * The default value is 0.8f.
       */
      public abstract Builder setTemperature(float temperature);

      /** Configures random seed for sampling tokens. */
      public abstract Builder setRandomSeed(int randomSeed);

      /**
       * The absolute path to the LoRA model asset bundle stored locally on the device. This is only
       * compatible with GPU models.
       */
      public abstract Builder setLoraPath(String loraPath);

      abstract LlmInferenceSessionOptions autoBuild();

      /** Validates and builds the {@link LlmInferenceSessionOptions} instance. */
      public final LlmInferenceSessionOptions build() {
        return autoBuild();
      }
    }

    /**
     * Top K number of tokens to be sampled from for each decoding step. A value of 1 means greedy
     * decoding.
     */
    public abstract int topK();

    /** Randomness when decoding the next token. A value of 0.0f means greedy decoding. */
    public abstract float temperature();

    /** Random seed for sampling tokens. */
    public abstract int randomSeed();

    /**
     * The absolute path to the LoRA model asset bundle stored locally on the device. This is only
     * compatible with GPU models.
     */
    public abstract Optional<String> loraPath();

    /** Instantiates a new LlmInferenceOptions builder. */
    public static Builder builder() {
      return new AutoValue_LlmInferenceSession_LlmInferenceSessionOptions.Builder()
          .setTopK(40)
          .setTemperature(0.8f)
          .setRandomSeed(0);
    }
  }

  static String stripLeading(String text) {
    // stripLeading() implementation for Android < 33
    int left = 0;
    while (left < text.length()) {
      final int codepoint = text.codePointAt(left);
      if (!Character.isWhitespace(codepoint)) {
        break;
      }
      left += Character.charCount(codepoint);
    }
    return text.substring(left);
  }
}
