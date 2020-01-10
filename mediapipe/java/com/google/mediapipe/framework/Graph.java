// Copyright 2019 The MediaPipe Authors.
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

package com.google.mediapipe.framework;

import com.google.common.base.Preconditions;
import com.google.common.flogger.FluentLogger;
import com.google.mediapipe.proto.CalculatorProto.CalculatorGraphConfig;
import com.google.mediapipe.proto.GraphTemplateProto.CalculatorGraphTemplate;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * MediaPipe-related context.
 *
 * <p>Main purpose is to faciliate the memory management for native allocated mediapipe objects.
 */
public class Graph {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();
  private static final int MAX_BUFFER_SIZE = 20;
  private long nativeGraphHandle;
  // Hold the references to callbacks.
  private final List<PacketCallback> packetCallbacks = new ArrayList<>();
  // Side packets used for running the graph.
  private Map<String, Packet> sidePackets = new HashMap<>();
  // Stream headers used for running the graph.
  private Map<String, Packet> streamHeaders = new HashMap<>();
  // The mode of running used by this context.
  // Based on the value of this mode, the caller can use {@link waitUntilIdle} to synchronize with
  // the mediapipe native graph runner.
  private boolean stepMode = false;

  private boolean startRunningGraphCalled = false;
  private boolean graphRunning = false;

  /** Helper class for a buffered Packet and its timestamp. */
  private static class PacketBufferItem {
    private PacketBufferItem(Packet packet, Long timestamp) {
      this.packet = packet;
      this.timestamp = timestamp;
    }

    final Packet packet;
    final Long timestamp;
  }

  private Map<String, ArrayList<PacketBufferItem>> packetBuffers = new HashMap<>();

  // This is used for methods that need to ensure the native context is alive
  // while still allowing other methods of this class to execute concurrently.
  // Note: if a method needs to acquire both this lock and the Graph intrinsic monitor,
  // it must acquire the intrinsic monitor first.
  private final Object terminationLock = new Object();

  public Graph() {
    nativeGraphHandle = nativeCreateGraph();
  }

  public synchronized long getNativeHandle() {
    return nativeGraphHandle;
  }

  public synchronized void setStepMode(boolean stepMode) {
    this.stepMode = stepMode;
  }

  public synchronized boolean getStepMode() {
    return stepMode;
  }

  /**
   * Loads a binary mediapipe graph using an absolute file path.
   *
   * @param path An absolute file path to a mediapipe graph. An absolute file path can be obtained
   *     from asset file using {@link AssetCache}.
   */
  public synchronized void loadBinaryGraph(String path) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    nativeLoadBinaryGraph(nativeGraphHandle, path);
  }

  /** Loads a binary mediapipe graph from a byte array. */
  public synchronized void loadBinaryGraph(byte[] data) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    nativeLoadBinaryGraphBytes(nativeGraphHandle, data);
  }

  /** Specifies a CalculatorGraphConfig for a mediapipe graph or subgraph. */
  public synchronized void loadBinaryGraph(CalculatorGraphConfig config) {
    loadBinaryGraph(config.toByteArray());
  }

  /** Specifies a CalculatorGraphTemplate for a mediapipe graph or subgraph. */
  public synchronized void loadBinaryGraphTemplate(CalculatorGraphTemplate template) {
    nativeLoadBinaryGraphTemplate(nativeGraphHandle, template.toByteArray());
  }

  /** Specifies the CalculatorGraphConfig::type of the top level graph. */
  public synchronized void setGraphType(String graphType) {
    nativeSetGraphType(nativeGraphHandle, graphType);
  }

  /** Specifies options such as template arguments for the graph. */
  public synchronized void setGraphOptions(CalculatorGraphConfig.Node options) {
    nativeSetGraphOptions(nativeGraphHandle, options.toByteArray());
  }

  /**
   * Returns the canonicalized CalculatorGraphConfig with subgraphs and graph templates expanded.
   */
  public synchronized CalculatorGraphConfig getCalculatorGraphConfig() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    byte[] data = nativeGetCalculatorGraphConfig(nativeGraphHandle);
    if (data != null) {
      try {
        return CalculatorGraphConfig.parseFrom(data);
      } catch (InvalidProtocolBufferException e) {
        throw new RuntimeException(e);
      }
    }
    return null;
  }

  /**
   * Adds a {@link PacketCallback} to the context for callback during graph running.
   *
   * @param streamName The output stream name in the graph for callback.
   * @param callback The callback for handling the call when output stream gets a {@link Packet}.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void addPacketCallback(String streamName, PacketCallback callback) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    Preconditions.checkNotNull(streamName);
    Preconditions.checkNotNull(callback);
    Preconditions.checkState(!graphRunning && !startRunningGraphCalled);
    packetCallbacks.add(callback);
    nativeAddPacketCallback(nativeGraphHandle, streamName, callback);
  }

  /**
   * Adds a {@link SurfaceOutput} for a stream producing GpuBuffers.
   *
   * <p>Multiple outputs can be attached to the same stream.
   *
   * @param streamName The output stream name in the graph.
   * @result a new SurfaceOutput.
   */
  public synchronized SurfaceOutput addSurfaceOutput(String streamName) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    Preconditions.checkNotNull(streamName);
    Preconditions.checkState(!graphRunning && !startRunningGraphCalled);
    // TODO: check if graph is loaded.
    return new SurfaceOutput(
        this, Packet.create(nativeAddSurfaceOutput(nativeGraphHandle, streamName)));
  }

  /**
   * Sets the input side packets needed for running the graph.
   *
   * @param sidePackets MediaPipe input side packet name to {@link Packet} map.
   */
  public synchronized void setInputSidePackets(Map<String, Packet> sidePackets) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    Preconditions.checkState(!graphRunning && !startRunningGraphCalled);
    for (Map.Entry<String, Packet> entry : sidePackets.entrySet()) {
      this.sidePackets.put(entry.getKey(), entry.getValue().copy());
    }
  }

  public synchronized <T> void setServiceObject(GraphService<T> service, T object) {
    service.installServiceObject(nativeGraphHandle, object);
  }

  /**
   * This tells the {@link Graph} before running the graph, we are expecting those headers to be
   * available first. This function is usually called before the streaming starts.
   *
   * <p>Note: Because of some MediaPipe calculators need statically available header info before the
   * graph is running, we need to have this to synchronize the running of graph with the
   * availability of the header streams.
   */
  public synchronized void addStreamNameExpectingHeader(String streamName) {
    Preconditions.checkState(!graphRunning && !startRunningGraphCalled);
    streamHeaders.put(streamName, null);
  }

  /**
   * Sets the stream header for specific stream if the header is not set.
   *
   * <p>If graph is already waiting for being started, start graph when all stream headers are set.
   *
   * <p>Note: If streamHeader is already being set, this call will not override the previous set
   * value. To override, call the function below instead.
   */
  public synchronized void setStreamHeader(String streamName, Packet streamHeader) {
    setStreamHeader(streamName, streamHeader, false);
  }

  /**
   * Sets the stream header for specific stream.
   *
   * <p>If graph is already waiting for being started, start graph when all stream headers are set.
   *
   * @param override if true, override the previous set header, however, if graph is running, {@link
   *     IllegalArgumentException} will be thrown.
   */
  public synchronized void setStreamHeader(
      String streamName, Packet streamHeader, boolean override) {
    Packet header = streamHeaders.get(streamName);
    if (header != null) {
      if (override) {
        if (graphRunning) {
          throw new IllegalArgumentException(
              "Can't override an existing stream header, after graph started running.");
        }
        header.release();
      } else {
        // Don't override, so just return since header is set already.
        return;
      }
    }
    streamHeaders.put(streamName, streamHeader.copy());
    if (!graphRunning && startRunningGraphCalled && hasAllStreamHeaders()) {
      startRunningGraph();
    }
  }

  /**
   * Runs the mediapipe graph until it finishes.
   *
   * <p>Side packets that are needed by the graph should be set using {@link setInputSidePackets}.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void runGraphUntilClose() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    Preconditions.checkNotNull(sidePackets);
    String[] streamNames = new String[sidePackets.size()];
    long[] packets = new long[sidePackets.size()];
    splitStreamNamePacketMap(sidePackets, streamNames, packets);
    nativeRunGraphUntilClose(nativeGraphHandle, streamNames, packets);
  }

  /**
   * Starts running the MediaPipe graph.
   *
   * <p>Returns immediately after starting the scheduler.
   *
   * <p>Side packets that are needed by the graph should be set using {@link setInputSidePackets}.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void startRunningGraph() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    startRunningGraphCalled = true;
    if (!hasAllStreamHeaders()) {
      // Graph will be runned later when all stream headers are assembled.
      logger.atInfo().log("MediaPipe graph won't start until all stream headers are available.");
      return;
    }
    // Prepare the side packets.
    String[] sidePacketNames = new String[sidePackets.size()];
    long[] sidePacketHandles = new long[sidePackets.size()];
    splitStreamNamePacketMap(sidePackets, sidePacketNames, sidePacketHandles);
    // Prepare the Stream headers.
    String[] streamNamesWithHeader = new String[streamHeaders.size()];
    long[] streamHeaderHandles = new long[streamHeaders.size()];
    splitStreamNamePacketMap(streamHeaders, streamNamesWithHeader, streamHeaderHandles);
    nativeStartRunningGraph(
        nativeGraphHandle,
        sidePacketNames,
        sidePacketHandles,
        streamNamesWithHeader,
        streamHeaderHandles);
    // Packets can be buffered before the actual mediapipe graph starts. Send them in now, if we
    // started successfully.
    graphRunning = true;
    moveBufferedPacketsToInputStream();
  }

  /**
   * Sets blocking behavior when adding packets to a graph input stream via {@link
   * addPacketToInputStream}. If set to true, the method will block until all dependent input
   * streams fall below the maximum queue size set in the graph config. If false, it will return and
   * not add a packet if any dependent input stream is full. To add a packet unconditionally, set
   * the maximum queue size to -1 in the graph config.
   */
  public synchronized void setGraphInputStreamBlockingMode(boolean mode) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    Preconditions.checkState(!graphRunning);
    nativeSetGraphInputStreamBlockingMode(nativeGraphHandle, mode);
  }

  /**
   * Adds one packet into a graph input stream based on the graph stream input mode.
   *
   * @param streamName the name of the input stream.
   * @param packet the mediapipe packet.
   * @param timestamp the timestamp of the packet, although not enforced, the unit is normally
   *     microsecond.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void addPacketToInputStream(
      String streamName, Packet packet, long timestamp) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    if (!graphRunning) {
      addPacketToBuffer(streamName, packet.copy(), timestamp);
    } else {
      nativeAddPacketToInputStream(
          nativeGraphHandle, streamName, packet.getNativeHandle(), timestamp);
    }
  }

  /**
   * Adds one packet into a graph input stream based on the graph stream input mode. Also
   * simultaneously yields ownership over to the graph stream, so additional memory optimizations
   * are possible. When the function ends normally, the packet will be consumed and should no longer
   * be referenced. When the function ends with MediaPipeException, the packet will remain
   * unaffected, so this call may be retried later.
   *
   * @param streamName the name of the input stream.
   * @param packet the mediapipe packet.
   * @param timestamp the timestamp of the packet, although not enforced, the unit is normally
   *     microsecond.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void addConsumablePacketToInputStream(
      String streamName, Packet packet, long timestamp) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    if (!graphRunning) {
      addPacketToBuffer(streamName, packet.copy(), timestamp);
      // Release current packet to honor move semantics.
      packet.release();
    } else {

      // We move the packet here into native, allowing it to take full control.
      nativeMovePacketToInputStream(
          nativeGraphHandle, streamName, packet.getNativeHandle(), timestamp);
      // The Java handle is released now if the packet was successfully moved. Otherwise the Java
      // handle continues to own the packet contents.
      packet.release();
    }
  }

  /**
   * Closes the specified input stream.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void closeInputStream(String streamName) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    nativeCloseInputStream(nativeGraphHandle, streamName);
  }

  /**
   * Closes all the input streams in the mediapipe graph.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void closeAllInputStreams() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    nativeCloseAllInputStreams(nativeGraphHandle);
  }

  /**
   * Closes all the input streams and source calculators in the mediapipe graph.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void closeAllPacketSources() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    nativeCloseAllPacketSources(nativeGraphHandle);
  }

  /**
   * Waits until the graph is done processing.
   *
   * <p>This should be called after all sources and input streams are closed.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void waitUntilGraphDone() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    nativeWaitUntilGraphDone(nativeGraphHandle);
  }

  /**
   * Waits until the graph runner is idle.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void waitUntilGraphIdle() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called.");
    nativeWaitUntilGraphIdle(nativeGraphHandle);
  }

  /** Releases the native mediapipe context. */
  public synchronized void tearDown() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    for (Map.Entry<String, Packet> entry : sidePackets.entrySet()) {
      entry.getValue().release();
    }
    sidePackets.clear();
    for (Map.Entry<String, Packet> entry : streamHeaders.entrySet()) {
      if (entry.getValue() != null) {
        entry.getValue().release();
      }
    }
    streamHeaders.clear();
    for (Map.Entry<String, ArrayList<PacketBufferItem>> entry : packetBuffers.entrySet()) {
      for (PacketBufferItem item : entry.getValue()) {
        item.packet.release();
      }
    }
    packetBuffers.clear();
    synchronized (terminationLock) {
      if (nativeGraphHandle != 0) {
        nativeReleaseGraph(nativeGraphHandle);
        nativeGraphHandle = 0;
      }
    }
    packetCallbacks.clear();
  }

  /**
   * Updates the value of a MediaPipe packet that holds a reference to another MediaPipe packet.
   *
   * <p>This updates a mutable packet. Useful for the caluclator that needs to have an external way
   * of updating the parameters using input side packets.
   *
   * <p>After calling this, the newPacket can be released (calling newPacket.release()), if no
   * longer need to use it in Java. The {@code referencePacket} already holds the reference.
   *
   * @param referencePacket a mediapipe packet that has the value type Packet*.
   * @param newPacket the new value for the reference packet to hold.
   */
  public synchronized void updatePacketReference(Packet referencePacket, Packet newPacket) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    nativeUpdatePacketReference(
        referencePacket.getNativeHandle(), newPacket.getNativeHandle());
  }

  /**
   * Creates a shared GL runner with the specified name so that MediaPipe calculators can use
   * OpenGL. This runner should be connected to the calculators by specifiying an input side packet
   * in the graph file with the same name.
   *
   * @throws MediaPipeException for any error status.
   * @deprecated Call {@link setParentGlContext} to set up texture sharing between contexts. Apart
   *     from that, GL is set up automatically.
   */
  @Deprecated
  public synchronized void createGlRunner(String name, long javaGlContext) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    Preconditions.checkArgument(name.equals("gpu_shared"));
    setParentGlContext(javaGlContext);
  }

  /**
   * Specifies an external GL context to use as the parent of MediaPipe's GL context. This will
   * enable the sharing of textures and other objects between the two contexts.
   *
   * <p>Cannot be called after the graph has been started.
   * @throws MediaPipeException for any error status.
   */
  public synchronized void setParentGlContext(long javaGlContext) {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    Preconditions.checkState(!graphRunning);
    nativeSetParentGlContext(nativeGraphHandle, javaGlContext);
  }

  /**
   * Cancels the running graph.
   */
  public synchronized void cancelGraph() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    nativeCancelGraph(nativeGraphHandle);
  }

  /** Returns {@link GraphProfiler}. */
  public GraphProfiler getProfiler() {
    Preconditions.checkState(
        nativeGraphHandle != 0, "Invalid context, tearDown() might have been called already.");
    return new GraphProfiler(nativeGetProfiler(nativeGraphHandle), this);
  }

  private boolean addPacketToBuffer(String streamName, Packet packet, long timestamp) {
    if (!packetBuffers.containsKey(streamName)) {
      packetBuffers.put(streamName, new ArrayList<PacketBufferItem>());
    }
    List<PacketBufferItem> buffer = packetBuffers.get(streamName);
    if (buffer.size() > MAX_BUFFER_SIZE) {
      for (Map.Entry<String, Packet> entry : streamHeaders.entrySet()) {
        if (entry.getValue() == null) {
          logger.atSevere().log("Stream: %s might be missing.", entry.getKey());
        }
      }
      throw new RuntimeException("Graph is not started because of missing streams");
    }
    buffer.add(new PacketBufferItem(packet, timestamp));
    return true;
  }

  // Any previously-buffered packets should be passed along to our graph.  They've already been
  // copied into our buffers, so it's fine to move them all over to native.
  private void moveBufferedPacketsToInputStream() {
    if (!packetBuffers.isEmpty()) {
      for (Map.Entry<String, ArrayList<PacketBufferItem>> entry : packetBuffers.entrySet()) {
        for (PacketBufferItem item : entry.getValue()) {
          try {
            nativeMovePacketToInputStream(
                nativeGraphHandle, entry.getKey(), item.packet.getNativeHandle(), item.timestamp);
          } catch (MediaPipeException e) {
            logger.atSevere().log(
                "AddPacket for stream: %s failed: %s.", entry.getKey(), e.getMessage());
            throw e;
          }
          // Need to release successfully moved packets
          item.packet.release();
        }
      }
      packetBuffers.clear();
    }
  }

  private static void splitStreamNamePacketMap(
      Map<String, Packet> namePacketMap, String[] streamNames, long[] packets) {
    if (namePacketMap.size() != streamNames.length || namePacketMap.size() != packets.length) {
      throw new RuntimeException("Input array length doesn't match the map size!");
    }
    int i = 0;
    for (Map.Entry<String, Packet> entry : namePacketMap.entrySet()) {
      streamNames[i] = entry.getKey();
      packets[i] = entry.getValue().getNativeHandle();
      ++i;
    }
  }

  private boolean hasAllStreamHeaders() {
    for (Map.Entry<String, Packet> entry : streamHeaders.entrySet()) {
      if (entry.getValue() == null) {
        return false;
      }
    }
    return true;
  }

  private native long nativeCreateGraph();

  private native void nativeReleaseGraph(long context);

  private native void nativeAddPacketCallback(
      long context, String streamName, PacketCallback callback);

  private native long nativeAddSurfaceOutput(long context, String streamName);

  private native void nativeLoadBinaryGraph(long context, String path);

  private native void nativeLoadBinaryGraphBytes(long context, byte[] data);

  private native void nativeLoadBinaryGraphTemplate(long context, byte[] data);

  private native void nativeSetGraphType(long context, String graphType);

  private native void nativeSetGraphOptions(long context, byte[] data);

  private native byte[] nativeGetCalculatorGraphConfig(long context);

  private native void nativeRunGraphUntilClose(long context, String[] streamNames, long[] packets);

  private native void nativeStartRunningGraph(
      long context,
      String[] sidePacketNames,
      long[] sidePacketHandles,
      String[] streamNamesWithHeader,
      long[] streamHeaderHandles);

  private native void nativeAddPacketToInputStream(
      long context, String streamName, long packet, long timestamp);

  private native void nativeMovePacketToInputStream(
      long context, String streamName, long packet, long timestamp);

  private native void nativeSetGraphInputStreamBlockingMode(long context, boolean mode);

  private native void nativeCloseInputStream(long context, String streamName);

  private native void nativeCloseAllInputStreams(long context);

  private native void nativeCloseAllPacketSources(long context);

  private native void nativeWaitUntilGraphDone(long context);

  private native void nativeWaitUntilGraphIdle(long context);

  private native void nativeUpdatePacketReference(long referencePacket, long newPacket);

  private native void nativeSetParentGlContext(long context, long javaGlContext);

  private native void nativeCancelGraph(long context);

  private native long nativeGetProfiler(long context);
}
