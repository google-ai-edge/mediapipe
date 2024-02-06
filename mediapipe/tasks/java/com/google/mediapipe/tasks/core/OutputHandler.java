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

package com.google.mediapipe.tasks.core;

import android.util.Log;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import java.util.List;

/** Base class for handling MediaPipe task graph outputs. */
public class OutputHandler<OutputT extends TaskResult, InputT> {
  /**
   * Interface for converting MediaPipe graph output {@link Packet}s to task result object and task
   * input object.
   */
  public interface OutputPacketConverter<OutputT extends TaskResult, InputT> {
    OutputT convertToTaskResult(List<Packet> packets);

    InputT convertToTaskInput(List<Packet> packets);
  }

  /**
   * Interface for the customizable MediaPipe task result listener that can retrieve both task
   * result objects and the corresponding input data.
   */
  public interface ResultListener<OutputT extends TaskResult, InputT> {
    void run(OutputT result, InputT input);
  }

  /**
   * Interface for the customizable MediaPipe task result listener that can only retrieve task
   * result objects.
   */
  public interface PureResultListener<OutputT extends TaskResult> {
    void run(OutputT result);
  }

  /**
   * Interface for the customizable MediaPipe task result listener that only receives a task's
   * output value.
   */
  public interface ValueListener<OutputT> {
    void run(OutputT result);
  }

  /**
   * Interface for the customizable MediaPipe task result listener that receives partial task
   * updates until it is invoked with `done` set to {@code true}.
   */
  public interface ProgressListener<OutputT> {
    void run(OutputT partialResult, boolean done);
  }

  private static final String TAG = "OutputHandler";
  // A task-specific graph output packet converter that should be implemented per task.
  private OutputPacketConverter<OutputT, InputT> outputPacketConverter;
  // The user-defined task result listener.
  private ResultListener<OutputT, InputT> resultListener;
  // The user-defined error listener.
  protected ErrorListener errorListener;
  // The cached task result for non latency sensitive use cases.
  protected OutputT cachedTaskResult;
  // The latest output timestamp.
  protected long latestOutputTimestamp = -1;
  // Whether the output handler should react to timestamp-bound changes by outputting empty packets.
  private boolean handleTimestampBoundChanges = false;

  /**
   * Sets a callback to be invoked to convert a {@link Packet} list to a task result object and a
   * task input object.
   *
   * @param converter the task-specific {@link OutputPacketConverter} callback.
   */
  public void setOutputPacketConverter(OutputPacketConverter<OutputT, InputT> converter) {
    this.outputPacketConverter = converter;
  }

  /**
   * Sets a callback to be invoked when task result objects become available.
   *
   * @param listener the user-defined {@link ResultListener} callback.
   */
  public void setResultListener(ResultListener<OutputT, InputT> listener) {
    this.resultListener = listener;
  }

  /**
   * Sets a callback to be invoked when exceptions are thrown from the task graph.
   *
   * @param listener The user-defined {@link ErrorListener} callback.
   */
  public void setErrorListener(ErrorListener listener) {
    this.errorListener = listener;
  }

  /**
   * Sets whether the output handler should react to the timestamp bound changes that are
   * represented as empty output {@link Packet}s.
   *
   * @param handleTimestampBoundChanges A boolean value.
   */
  public void setHandleTimestampBoundChanges(boolean handleTimestampBoundChanges) {
    this.handleTimestampBoundChanges = handleTimestampBoundChanges;
  }

  /** Returns true if the task graph is set to handle timestamp bound changes. */
  boolean handleTimestampBoundChanges() {
    return handleTimestampBoundChanges;
  }

  /* Returns the cached task result object. */
  public OutputT retrieveCachedTaskResult() {
    OutputT taskResult = cachedTaskResult;
    cachedTaskResult = null;
    return taskResult;
  }

  /* Returns the latest output timestamp. */
  public long getLatestOutputTimestamp() {
    return latestOutputTimestamp;
  }

  /**
   * Handles a list of output {@link Packet}s. Invoked when a packet list become available.
   *
   * @param packets A list of output {@link Packet}s.
   */
  void run(List<Packet> packets) {
    OutputT taskResult = null;
    try {
      taskResult = outputPacketConverter.convertToTaskResult(packets);
      if (resultListener == null) {
        cachedTaskResult = taskResult;
        latestOutputTimestamp = packets.get(0).getTimestamp();
      } else {
        InputT taskInput = outputPacketConverter.convertToTaskInput(packets);
        resultListener.run(taskResult, taskInput);
      }
    } catch (MediaPipeException e) {
      if (errorListener != null) {
        errorListener.onError(e);
      } else {
        Log.e(TAG, "Error occurs when getting MediaPipe task result. " + e);
      }
    }
  }
}
