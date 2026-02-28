package com.google.mediapipe.tasks.genai.llminference;

/** Delegate method for the LlmTaskRunner that handles async responses from the JNI layer. */
interface LlmTaskRunnerDelegate {

  /**
   * Handles an async response from the JNI layer.
   *
   * @param responseBytes The LlmResponseContext proto in bytes.
   */
  void onAsyncResponse(byte[] responseBytes);
}
