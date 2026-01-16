// Copyright 2021 The MediaPipe Authors.
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

package com.google.mediapipe.solutioncore;

import static java.util.concurrent.TimeUnit.MICROSECONDS;
import static java.util.concurrent.TimeUnit.MILLISECONDS;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;
import com.google.common.collect.ImmutableList;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Graph;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.solutioncore.logging.SolutionStatsDummyLogger;
import com.google.mediapipe.solutioncore.logging.SolutionStatsLogger;
import com.google.protobuf.Parser;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import javax.annotation.Nullable;

/** The base class of the MediaPipe solutions. */
public class SolutionBase implements AutoCloseable {
  private static final String TAG = "SolutionBase";
  protected Graph solutionGraph;
  protected AndroidPacketCreator packetCreator;
  protected ErrorListener errorListener;
  protected String imageInputStreamName;
  protected long lastTimestamp = Long.MIN_VALUE;
  protected final AtomicBoolean solutionGraphStarted = new AtomicBoolean(false);
  protected SolutionStatsLogger statsLogger;

  static {
    System.loadLibrary("mediapipe_jni");
  }

  /**
   * Initializes solution base with Android context, solution specific settings, and solution result
   * handler.
   *
   * @param context an Android {@link Context}.
   * @param solutionInfo a {@link SolutionInfo} contains binary graph file path, graph input and
   *     output stream names.
   * @param outputHandler a {@link OutputHandler} handles both solution result object and runtime
   *     exception.
   */
  public synchronized void initialize(
      Context context,
      SolutionInfo solutionInfo,
      OutputHandler<? extends SolutionResult> outputHandler) {
    this.imageInputStreamName = solutionInfo.imageInputStreamName();
    this.statsLogger =
        new SolutionStatsDummyLogger(context, this.getClass().getSimpleName(), solutionInfo);
    try {
      AndroidAssetUtil.initializeNativeAssetManager(context);
      solutionGraph = new Graph();
      if (new File(solutionInfo.binaryGraphPath()).isAbsolute()) {
        solutionGraph.loadBinaryGraph(solutionInfo.binaryGraphPath());
      } else {
        solutionGraph.loadBinaryGraph(
            AndroidAssetUtil.getAssetBytes(context.getAssets(), solutionInfo.binaryGraphPath()));
      }
      outputHandler.setStatsLogger(statsLogger);
      solutionGraph.addMultiStreamCallback(
          solutionInfo.outputStreamNames(),
          outputHandler::run,
          /*observeTimestampBounds=*/ outputHandler.handleTimestampBoundChanges());
      packetCreator = new AndroidPacketCreator(solutionGraph);
    } catch (MediaPipeException e) {
      statsLogger.logInitError();
      reportError("Error occurs while creating the MediaPipe solution graph.", e);
    }
  }

  /** Reports error with the detailed error message. */
  protected void reportError(String message, MediaPipeException e) {
    String detailedErrorMessage = String.format("%s Error details: %s", message, e.getMessage());
    if (errorListener != null) {
      errorListener.onError(detailedErrorMessage, e);
    } else {
      Log.e(TAG, detailedErrorMessage, e);
      throw e;
    }
  }

  /**
   * A convinence method to get proto list from a packet. If packet is empty, returns an empty list.
   */
  protected <T> List<T> getProtoVector(Packet packet, Parser<T> messageParser) {
    return packet.isEmpty()
        ? ImmutableList.<T>of()
        : PacketGetter.getProtoVector(packet, messageParser);
  }

  /** Gets current timestamp in microseconds. */
  protected long getCurrentTimestampUs() {
    return MICROSECONDS.convert(SystemClock.elapsedRealtime(), MILLISECONDS);
  }

  /** Starts the solution graph by taking an optional input side packets map. */
  public synchronized void start(@Nullable Map<String, Packet> inputSidePackets) {
    try {
      if (inputSidePackets != null) {
        solutionGraph.setInputSidePackets(inputSidePackets);
      }
      if (!solutionGraphStarted.get()) {
        solutionGraph.startRunningGraph();
        // Wait until all calculators are opened and the graph is truly started.
        solutionGraph.waitUntilGraphIdle();
        solutionGraphStarted.set(true);
        statsLogger.logSessionStart();
      }
    } catch (MediaPipeException e) {
      statsLogger.logInitError();
      reportError("Error occurs while starting the MediaPipe solution graph.", e);
    }
  }

  /** A blocking API that returns until the solution finishes processing all the pending tasks. */
  public void waitUntilIdle() {
    try {
      solutionGraph.waitUntilGraphIdle();
    } catch (MediaPipeException e) {
      reportError("Error occurs while waiting until the MediaPipe graph becomes idle.", e);
    }
  }

  /** Closes and cleans up the solution graph. */
  @Override
  public void close() {
    if (solutionGraphStarted.get()) {
      try {
        solutionGraph.closeAllPacketSources();
        solutionGraph.waitUntilGraphDone();
        statsLogger.logSessionEnd();
      } catch (MediaPipeException e) {
        // Note: errors during Process are reported at the earliest opportunity,
        // which may be addPacket or waitUntilDone, depending on timing. For consistency,
        // we want to always report them using the same async handler if installed.
        reportError("Error occurs while closing the Mediapipe solution graph.", e);
      }
      try {
        solutionGraph.tearDown();
      } catch (MediaPipeException e) {
        reportError("Error occurs while closing the Mediapipe solution graph.", e);
      }
    }
  }
}
