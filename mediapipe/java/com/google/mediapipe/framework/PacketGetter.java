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
import com.google.mediapipe.framework.PacketUtil.SerializedMessage;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLite;
import com.google.protobuf.Parser;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Converts the {@link Packet} to java accessible data types.
 *
 * <p>{@link Packet} is a thin java wrapper for the native MediaPipe packet. This class provides the
 * extendable conversion needed to access the data in the packet.
 *
 * <p>Note that it is still the developer's responsibility to interpret the data correctly.
 */
public final class PacketGetter {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  /** Helper class for a list of exactly two Packets. */
  public static class PacketPair {
    public PacketPair(Packet first, Packet second) {
      this.first = first;
      this.second = second;
    }

    final Packet first;
    final Packet second;
  }

  /**
   * Returns the {@link Packet} that held in the reference packet.
   *
   * <p>Note: release the returned packet after use.
   */
  public static Packet getPacketFromReference(final Packet referencePacket) {
    return Packet.create(nativeGetPacketFromReference(referencePacket.getNativeHandle()));
  }

  /**
   * The {@link Packet} contains a pair of packets, return both of them.
   *
   * <p>Note: release the packets in the pair after use.
   *
   * @param packet A MediaPipe packet that contains a pair of packets.
   */
  public static PacketPair getPairOfPackets(final Packet packet) {
     long[] handles = nativeGetPairPackets(packet.getNativeHandle());
    return new PacketPair(Packet.create(handles[0]), Packet.create(handles[1]));
  }

  /**
   * Returns a list of packets that are contained in The {@link Packet}.
   *
   * <p>Note: release the packets in the list after use.
   *
   * @param packet A MediaPipe packet that contains a vector of packets.
   */
  public static List<Packet> getVectorOfPackets(final Packet packet) {
     long[] handles = nativeGetVectorPackets(packet.getNativeHandle());
    List<Packet> packets = new ArrayList<>(handles.length);
     for (long handle : handles) {
      packets.add(Packet.create(handle));
     }
     return packets;
  }

  public static short getInt16(final Packet packet) {
    return nativeGetInt16(packet.getNativeHandle());
  }

  public static int getInt32(final Packet packet) {
    return nativeGetInt32(packet.getNativeHandle());
  }

  public static long getInt64(final Packet packet) {
    return nativeGetInt64(packet.getNativeHandle());
  }

  public static float getFloat32(final Packet packet) {
    return nativeGetFloat32(packet.getNativeHandle());
  }

  public static double getFloat64(final Packet packet) {
    return nativeGetFloat64(packet.getNativeHandle());
  }

  public static boolean getBool(final Packet packet) {
    return nativeGetBool(packet.getNativeHandle());
  }

  public static String getString(final Packet packet) {
    return nativeGetString(packet.getNativeHandle());
  }

  public static byte[] getBytes(final Packet packet) {
    return nativeGetBytes(packet.getNativeHandle());
  }

  public static byte[] getProtoBytes(final Packet packet) {
    return nativeGetProtoBytes(packet.getNativeHandle());
  }

  public static <T extends MessageLite> T getProto(final Packet packet, Class<T> clazz)
      throws InvalidProtocolBufferException {
    SerializedMessage result = new SerializedMessage();
    nativeGetProto(packet.getNativeHandle(), result);
    return PacketUtil.unpack(result, clazz);
  }

  public static short[] getInt16Vector(final Packet packet) {
    return nativeGetInt16Vector(packet.getNativeHandle());
  }

  public static int[] getInt32Vector(final Packet packet) {
    return nativeGetInt32Vector(packet.getNativeHandle());
  }

  public static long[] getInt64Vector(final Packet packet) {
    return nativeGetInt64Vector(packet.getNativeHandle());
  }

  public static float[] getFloat32Vector(final Packet packet) {
    return nativeGetFloat32Vector(packet.getNativeHandle());
  }

  public static double[] getFloat64Vector(final Packet packet) {
    return nativeGetFloat64Vector(packet.getNativeHandle());
  }

  public static <T> List<T> getProtoVector(final Packet packet, Parser<T> messageParser) {
    byte[][] protoVector = nativeGetProtoVector(packet.getNativeHandle());
    Preconditions.checkNotNull(
        protoVector, "Vector of protocol buffer objects should not be null!");
    try {
      List<T> parsedMessageList = new ArrayList<>();
      for (byte[] message : protoVector) {
        T parsedMessage = messageParser.parseFrom(message);
        parsedMessageList.add(parsedMessage);
      }
      return parsedMessageList;
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public static int getImageWidth(final Packet packet) {
    return nativeGetImageWidth(packet.getNativeHandle());
  }

  public static int getImageHeight(final Packet packet) {
    return nativeGetImageHeight(packet.getNativeHandle());
  }

  /**
   * Returns the native image buffer in ByteBuffer. It assumes the output buffer stores pixels
   * contiguously. It returns false if this assumption does not hold.
   *
   * <p>Note: this function does not assume the pixel format.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public static boolean getImageData(final Packet packet, ByteBuffer buffer) {
    return nativeGetImageData(packet.getNativeHandle(), buffer);
  }

  /**
   * Converts an RGB mediapipe image frame packet to an RGBA Byte buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public static boolean getRgbaFromRgb(final Packet packet, ByteBuffer buffer) {
    return nativeGetRgbaFromRgb(packet.getNativeHandle(), buffer);
  }

  /**
   * Converts the audio matrix data back into byte data.
   *
   * <p>The matrix is in column major order.
   */
  public static byte[] getAudioByteData(final Packet packet) {
    return nativeGetAudioData(packet.getNativeHandle());
  }

  /**
   * Audio data is in MediaPipe Matrix format.
   *
   * @return the number of channels in the data.
   */
  public static int getAudioDataNumChannels(final Packet packet) {
    return nativeGetMatrixRows(packet.getNativeHandle());
  }

  /**
   * Audio data is in MediaPipe Matrix format.
   *
   * @return the number of samples in the data.
   */
  public static int getAudioDataNumSamples(final Packet packet) {
    return nativeGetMatrixCols(packet.getNativeHandle());
  }

  /**
   * In addition to the data packet, mediapipe currently also has a separate audio header: {@code
   * mediapipe::TimeSeriesHeader}.
   *
   * @return the number of channel in the header packet.
   */
  public static int getTimeSeriesHeaderNumChannels(final Packet packet) {
    return nativeGetTimeSeriesHeaderNumChannels(packet.getNativeHandle());
  }

  /**
   * In addition to the data packet, mediapipe currently also has a separate audio header: {@code
   * mediapipe::TimeSeriesHeader}.
   *
   * @return the sampling rate in the header packet.
   */
  public static double getTimeSeriesHeaderSampleRate(final Packet packet) {
    return nativeGetTimeSeriesHeaderSampleRate(packet.getNativeHandle());
  }

  /** Gets the width in video header packet. */
  public static int getVideoHeaderWidth(final Packet packet) {
    return nativeGetVideoHeaderWidth(packet.getNativeHandle());
  }

  /** Gets the height in video header packet. */
  public static int getVideoHeaderHeight(final Packet packet) {
    return nativeGetVideoHeaderHeight(packet.getNativeHandle());
  }

  /**
   * Returns the float array data of the mediapipe Matrix.
   *
   * <p>Underlying packet stores the matrix as {@code ::mediapipe::Matrix}.
   */
  public static float[] getMatrixData(final Packet packet) {
    return nativeGetMatrixData(packet.getNativeHandle());
  }

  public static int getMatrixRows(final Packet packet) {
    return nativeGetMatrixRows(packet.getNativeHandle());
  }

  public static int getMatrixCols(final Packet packet) {
    return nativeGetMatrixCols(packet.getNativeHandle());
  }

  /**
   * Returns the GL texture name of the mediapipe::GpuBuffer.
   *
   * @deprecated use {@link #getTextureFrame} instead.
   */
  @Deprecated
  public static int getGpuBufferName(final Packet packet) {
    return nativeGetGpuBufferName(packet.getNativeHandle());
  }

  /**
   * Returns a {@link GraphTextureFrame} referencing a C++ mediapipe::GpuBuffer.
   *
   * <p>Note: in order for the application to be able to use the texture, its GL context must be
   * linked with MediaPipe's. This is ensured by calling {@link Graph#createGlRunner(String,long)}
   * with the native handle to the application's GL context as the second argument.
   *
   * <p>The returned GraphTextureFrame must be released by the caller. If this method is called
   * multiple times, each returned GraphTextureFrame is an independent reference to the underlying
   * texture data, and must be released individually.
   */
  public static GraphTextureFrame getTextureFrame(final Packet packet) {
    return new GraphTextureFrame(
        nativeGetGpuBuffer(packet.getNativeHandle()), packet.getTimestamp());
  }

  private static native long nativeGetPacketFromReference(long nativePacketHandle);
  private static native long[] nativeGetPairPackets(long nativePacketHandle);
  private static native long[] nativeGetVectorPackets(long nativePacketHandle);

  private static native short nativeGetInt16(long nativePacketHandle);
  private static native int nativeGetInt32(long nativePacketHandle);
  private static native long nativeGetInt64(long nativePacketHandle);
  private static native float nativeGetFloat32(long nativePacketHandle);
  private static native double nativeGetFloat64(long nativePacketHandle);
  private static native boolean nativeGetBool(long nativePacketHandle);
  private static native String nativeGetString(long nativePacketHandle);
  private static native byte[] nativeGetBytes(long nativePacketHandle);
  private static native byte[] nativeGetProtoBytes(long nativePacketHandle);
  private static native void nativeGetProto(long nativePacketHandle, SerializedMessage result);
  private static native short[] nativeGetInt16Vector(long nativePacketHandle);
  private static native int[] nativeGetInt32Vector(long nativePacketHandle);
  private static native long[] nativeGetInt64Vector(long nativePacketHandle);
  private static native float[] nativeGetFloat32Vector(long nativePacketHandle);
  private static native double[] nativeGetFloat64Vector(long nativePacketHandle);

  private static native byte[][] nativeGetProtoVector(long nativePacketHandle);

  private static native int nativeGetImageWidth(long nativePacketHandle);
  private static native int nativeGetImageHeight(long nativePacketHandle);
  private static native boolean nativeGetImageData(long nativePacketHandle, ByteBuffer buffer);
  private static native boolean nativeGetRgbaFromRgb(long nativePacketHandle, ByteBuffer buffer);
  // Retrieves the values that are in the VideoHeader.
  private static native int nativeGetVideoHeaderWidth(long nativepackethandle);
  private static native int nativeGetVideoHeaderHeight(long nativepackethandle);
  // Retrieves the values that are in the mediapipe::TimeSeriesHeader.
  private static native int nativeGetTimeSeriesHeaderNumChannels(long nativepackethandle);

  private static native double nativeGetTimeSeriesHeaderSampleRate(long nativepackethandle);

  // Audio data in MediaPipe current uses MediaPipe Matrix format type.
  private static native byte[] nativeGetAudioData(long nativePacketHandle);
  // Native helper functions to access the MediaPipe Matrix data.
  private static native float[] nativeGetMatrixData(long nativePacketHandle);

  private static native int nativeGetMatrixRows(long nativePacketHandle);
  private static native int nativeGetMatrixCols(long nativePacketHandle);
  private static native int nativeGetGpuBufferName(long nativePacketHandle);
  private static native long nativeGetGpuBuffer(long nativePacketHandle);

  private PacketGetter() {}
}
