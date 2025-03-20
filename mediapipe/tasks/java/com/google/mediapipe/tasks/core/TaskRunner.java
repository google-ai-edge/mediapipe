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

import android.content.Context;
import android.util.Log;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Graph;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.tasks.core.logging.TasksStatsLogger;
import com.google.mediapipe.tasks.core.logging.TasksStatsLoggerFactory;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/** The runner of MediaPipe task graphs. */
public class TaskRunner implements AutoCloseable {
  private static final String TAG = TaskRunner.class.getSimpleName();
  private static final long TIMESATMP_UNITS_PER_SECOND = 1000000;

  private final OutputHandler<? extends TaskResult, ?> outputHandler;
  private final AtomicBoolean graphStarted = new AtomicBoolean(false);
  private final Graph graph;
  private final ModelResourcesCache modelResourcesCache;
  private final AndroidPacketCreator packetCreator;
  private final TasksStatsLogger statsLogger;
  private long lastSeenTimestamp = Long.MIN_VALUE;
  private ErrorListener errorListener;

  /**
   * Create a {@link TaskRunner} instance.
   *
   * @param context an Android {@link Context}.
   * @param taskInfo a {@link TaskInfo} instance contains task graph name, task options, and graph
   *     input and output stream names.
   * @param outputHandler a {@link OutputHandler} instance handles task result object and runtime
   *     exception.
   * @throws MediaPipeException for any error during {@link TaskRunner} creation.
   */
  public static TaskRunner create(
      Context context,
      TaskInfo<? extends TaskOptions> taskInfo,
      OutputHandler<? extends TaskResult, ?> outputHandler) {
    TasksStatsLogger statsLogger =
        TasksStatsLoggerFactory.create(
            context, taskInfo.taskName(), taskInfo.taskRunningModeName());
    AndroidAssetUtil.initializeNativeAssetManager(context);
    Graph mediapipeGraph = new Graph();
    mediapipeGraph.loadBinaryGraph(taskInfo.generateGraphConfig());
    ModelResourcesCache graphModelResourcesCache = new ModelResourcesCache();
    mediapipeGraph.setServiceObject(new ModelResourcesCacheService(), graphModelResourcesCache);
    mediapipeGraph.addMultiStreamCallback(
        taskInfo.outputStreamNames(),
        packets -> {
          outputHandler.run(packets);
          statsLogger.recordInvocationEnd(packets.get(0).getTimestamp());
        },
        /* observeTimestampBounds= */ outputHandler.handleTimestampBoundChanges());
    mediapipeGraph.startRunningGraph();
    // Waits until all calculators are opened and the graph is fully started.
    mediapipeGraph.waitUntilGraphIdle();
    return new TaskRunner(mediapipeGraph, graphModelResourcesCache, outputHandler, statsLogger);
  }

  /**
   * Sets a callback to be invoked when exceptions are thrown by the {@link TaskRunner} instance.
   *
   * @param listener an {@link ErrorListener} callback.
   */
  public void setErrorListener(ErrorListener listener) {
    this.errorListener = listener;
  }

  /** Returns the {@link AndroidPacketCreator} associated to the {@link TaskRunner} instance. */
  public AndroidPacketCreator getPacketCreator() {
    return packetCreator;
  }

  /**
   * A synchronous method for processing batch data.
   *
   * <p>Note: This method is designed for processing batch data such as unrelated images and texts.
   * The call blocks the current thread until a failure status or a successful result is returned.
   * An internal timestamp will be assigned per invocation. This method is thread-safe and allows
   * clients to call it from different threads.
   *
   * @param inputs a map contains (input stream {@link String}, data {@link Packet}) pairs.
   */
  public synchronized TaskResult process(Map<String, Packet> inputs) {
    long syntheticInputTimestamp = generateSyntheticTimestamp();
    // TODO: Support recording GPU input arrival.
    statsLogger.recordCpuInputArrival(syntheticInputTimestamp);
    addPackets(inputs, syntheticInputTimestamp);
    graph.waitUntilGraphIdle();
    lastSeenTimestamp = outputHandler.getLatestOutputTimestamp();
    return outputHandler.retrieveCachedTaskResult();
  }

  /**
   * A synchronous method for processing offline streaming data.
   *
   * <p>Note: This method is designed for processing offline streaming data such as the decoded
   * frames from a video file and an audio file. The call blocks the current thread until a failure
   * status or a successful result is returned. The caller must ensure that the input timestamp is
   * greater than the timestamps of previous invocations. This method is thread-unsafe and it is the
   * caller's responsibility to synchronize access to this method across multiple threads and to
   * ensure that the input packet timestamps are in order.
   *
   * @param inputs a map contains (input stream {@link String}, data {@link Packet}) pairs.
   * @param inputTimestamp the timestamp of the input packets.
   */
  public synchronized TaskResult process(Map<String, Packet> inputs, long inputTimestamp) {
    validateInputTimstamp(inputTimestamp);
    statsLogger.recordCpuInputArrival(inputTimestamp);
    addPackets(inputs, inputTimestamp);
    graph.waitUntilGraphIdle();
    return outputHandler.retrieveCachedTaskResult();
  }

  /**
   * An asynchronous method for handling live streaming data.
   *
   * <p>Note: This method that is designed for handling live streaming data such as live camera and
   * microphone data. A user-defined packets callback function must be provided in the constructor
   * to receive the output packets. The caller must ensure that the input packet timestamps are
   * monotonically increasing. This method is thread-unsafe and it is the caller's responsibility to
   * synchronize access to this method across multiple threads and to ensure that the input packet
   * timestamps are in order.
   *
   * @param inputs a map contains (input stream {@link String}, data {@link Packet}) pairs.
   * @param inputTimestamp the timestamp of the input packets.
   */
  public synchronized void send(Map<String, Packet> inputs, long inputTimestamp) {
    validateInputTimstamp(inputTimestamp);
    statsLogger.recordCpuInputArrival(inputTimestamp);
    addPackets(inputs, inputTimestamp);
  }

  /**
   * Resets and restarts the {@link TaskRunner} instance. This can be useful for resetting a
   * stateful task graph to process new data.
   */
  public void restart() {
    if (graphStarted.get()) {
      try {
        graphStarted.set(false);
        graph.closeAllPacketSources();
        graph.waitUntilGraphDone();
        statsLogger.logSessionEnd();
      } catch (MediaPipeException e) {
        reportError(e);
      }
    }
    try {
      graph.startRunningGraph();
      // Waits until all calculators are opened and the graph is fully restarted.
      graph.waitUntilGraphIdle();
      graphStarted.set(true);
      statsLogger.logSessionStart();
    } catch (MediaPipeException e) {
      reportError(e);
    }
  }

  /** Closes and cleans up the {@link TaskRunner} instance. */
  @Override
  public void close() {
    if (!graphStarted.get()) {
      return;
    }
    try {
      graphStarted.set(false);
      graph.closeAllPacketSources();
      graph.waitUntilGraphDone();
      statsLogger.logSessionEnd();
      if (modelResourcesCache != null) {
        modelResourcesCache.release();
      }
    } catch (MediaPipeException e) {
      // Note: errors during Process are reported at the earliest opportunity,
      // which may be addPacket or waitUntilDone, depending on timing. For consistency,
      // we want to always report them using the same async handler if installed.
      reportError(e);
    }
    try {
      graph.tearDown();
    } catch (MediaPipeException e) {
      reportError(e);
    }
  }

  public CalculatorGraphConfig getCalculatorGraphConfig() {
    return graph.getCalculatorGraphConfig();
  }

  private synchronized void addPackets(Map<String, Packet> inputs, long inputTimestamp) {
    if (!graphStarted.get()) {
      reportError(
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "The task graph hasn't been successfully started or error occurs during graph"
                  + " initializaton."));
    }
    try {
      for (Map.Entry<String, Packet> entry : inputs.entrySet()) {
        // addConsumablePacketToInputStream allows the graph to take exclusive ownership of the
        // packet, which may allow for more memory optimizations.
        graph.addConsumablePacketToInputStream(entry.getKey(), entry.getValue(), inputTimestamp);
        // If addConsumablePacket succeeded, we don't need to release the packet ourselves.
        entry.setValue(null);
      }
    } catch (MediaPipeException e) {
      // TODO: do not suppress exceptions here!
      if (errorListener == null) {
        Log.e(TAG, "Mediapipe error: ", e);
      } else {
        throw e;
      }
    } finally {
      for (Packet packet : inputs.values()) {
        // In case of error, addConsumablePacketToInputStream will not release the packet, so we
        // have to release it ourselves.
        if (packet != null) {
          packet.release();
        }
      }
    }
  }

  /**
   * Checks if the input timestamp is strictly greater than the last timestamp that has been
   * processed.
   *
   * @param inputTimestamp the input timestamp.
   */
  private void validateInputTimstamp(long inputTimestamp) {
    if (lastSeenTimestamp >= inputTimestamp) {
      reportError(
          new MediaPipeException(
              MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
              "The received packets having a smaller timestamp than the processed timestamp."));
    }
    lastSeenTimestamp = inputTimestamp;
  }

  /** Generates a synthetic input timestamp in the batch processing mode. */
  private long generateSyntheticTimestamp() {
    long timestamp =
        lastSeenTimestamp == Long.MIN_VALUE ? 0 : lastSeenTimestamp + TIMESATMP_UNITS_PER_SECOND;
    lastSeenTimestamp = timestamp;
    return timestamp;
  }

  /** Private constructor. */
  private TaskRunner(
      Graph graph,
      ModelResourcesCache modelResourcesCache,
      OutputHandler<? extends TaskResult, ?> outputHandler,
      TasksStatsLogger statsLogger) {
    this.outputHandler = outputHandler;
    this.graph = graph;
    this.modelResourcesCache = modelResourcesCache;
    this.packetCreator = new AndroidPacketCreator(graph);
    this.statsLogger = statsLogger;
    graphStarted.set(true);
    this.statsLogger.logSessionStart();
  }

  /** Reports error. */
  private void reportError(MediaPipeException e) {
    if (errorListener != null) {
      errorListener.onError(e);
    } else {
      throw e;
    }
  }
}
