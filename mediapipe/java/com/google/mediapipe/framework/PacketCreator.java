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

import com.google.mediapipe.framework.PacketUtil.SerializedMessage;
import com.google.protobuf.MessageLite;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

// TODO: use Preconditions in this file.
/**
 * Creates {@link Packet} in the given {@link Graph}.
 *
 * <p>This class provides a set of functions to create basic mediapipe packet types.
 */
public class PacketCreator {
  protected Graph mediapipeGraph;

  public PacketCreator(Graph context) {
    mediapipeGraph = context;
  }

  /**
   * Create a MediaPipe Packet that contains a pointer to another MediaPipe packet.
   *
   * <p>This can be used as a way to update the value of a packet. Similar to a mutable packet using
   * mediapipe::AdoptAsUniquePtr.
   *
   * <p>The parameter {@code packet} can be released after this call, since the new packet already
   * holds a reference to it in the native object.
   */
  public Packet createReferencePacket(Packet packet) {
    return Packet.create(
        nativeCreateReferencePacket(mediapipeGraph.getNativeHandle(), packet.getNativeHandle()));
  }

  /**
   * Creates a 3 channel RGB ImageFrame packet from an RGB buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer. The pixel rows should have
   * 4-byte alignment.
   */
  public Packet createRgbImage(ByteBuffer buffer, int width, int height) {
    int widthStep = (((width * 3) + 3) / 4) * 4;
    if (widthStep * height != buffer.capacity()) {
      throw new RuntimeException("The size of the buffer should be: " + widthStep * height);
    }
    return Packet.create(
        nativeCreateRgbImage(mediapipeGraph.getNativeHandle(), buffer, width, height));
  }

  /**
   * Create a MediaPipe audio packet that is used by most of the audio calculators.
   *
   * @param data the raw audio data, bytes per sample is 2.
   * @param numChannels number of channels in the raw data.
   * @param numSamples number of samples in the data.
   */
  public Packet createAudioPacket(byte[] data, int numChannels, int numSamples) {
    checkAudioDataSize(data.length, numChannels, numSamples);
    return Packet.create(
        nativeCreateAudioPacket(
            mediapipeGraph.getNativeHandle(), data, /*offset=*/ 0, numChannels, numSamples));
  }

  /**
   * Create a MediaPipe audio packet that is used by most of the audio calculators.
   *
   * @param data the raw audio data, bytes per sample is 2. Must either be a direct byte buffer or
   *     have an array.
   * @param numChannels number of channels in the raw data.
   * @param numSamples number of samples in the data.
   */
  public Packet createAudioPacket(ByteBuffer data, int numChannels, int numSamples) {
    checkAudioDataSize(data.remaining(), numChannels, numSamples);
    if (data.isDirect()) {
      return Packet.create(
          nativeCreateAudioPacketDirect(
              mediapipeGraph.getNativeHandle(), data.slice(), numChannels, numSamples));
    } else if (data.hasArray()) {
      return Packet.create(
          nativeCreateAudioPacket(
              mediapipeGraph.getNativeHandle(),
              data.array(),
              data.arrayOffset() + data.position(),
              numChannels,
              numSamples));
    } else {
      throw new IllegalArgumentException(
          "Data must be either a direct byte buffer or be backed by a byte array.");
    }
  }

  private static void checkAudioDataSize(int length, int numChannels, int numSamples) {
    final int expectedLength = numChannels * numSamples * 2;
    if (expectedLength != length) {
      throw new IllegalArgumentException(
          "Please check the audio data size, has to be num_channels * num_samples * 2 = "
              + expectedLength
              + " but was "
              + length);
    }
  }

  /**
   * Creates a 3 channel RGB ImageFrame packet from an RGBA buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public Packet createRgbImageFromRgba(ByteBuffer buffer, int width, int height) {
    if (width * height * 4 != buffer.capacity()) {
      throw new RuntimeException("The size of the buffer should be: " + width * height * 4);
    }
    return Packet.create(
        nativeCreateRgbImageFromRgba(mediapipeGraph.getNativeHandle(), buffer, width, height));
  }

  /**
   * Creates a 1 channel ImageFrame packet from an U8 buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public Packet createGrayscaleImage(ByteBuffer buffer, int width, int height) {
    if (width * height != buffer.capacity()) {
      throw new RuntimeException(
          "The size of the buffer should be: " + width * height + " but is " + buffer.capacity());
    }
    return Packet.create(
        nativeCreateGrayscaleImage(mediapipeGraph.getNativeHandle(), buffer, width, height));
  }

  /**
   * Creates a 4 channel RGBA ImageFrame packet from an RGBA buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public Packet createRgbaImageFrame(ByteBuffer buffer, int width, int height) {
    if (buffer.capacity() != width * height * 4) {
      throw new RuntimeException("buffer doesn't have the correct size.");
    }
    return Packet.create(
        nativeCreateRgbaImageFrame(mediapipeGraph.getNativeHandle(), buffer, width, height));
  }

  /**
   * Creates a 1 channel float ImageFrame packet.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   */
  public Packet createFloatImageFrame(FloatBuffer buffer, int width, int height) {
    if (buffer.capacity() != width * height * 4) {
      throw new RuntimeException("buffer doesn't have the correct size.");
    }
    return Packet.create(
        nativeCreateFloatImageFrame(mediapipeGraph.getNativeHandle(), buffer, width, height));
  }

  public Packet createInt16(short value) {
    return Packet.create(nativeCreateInt16(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createInt32(int value) {
    return Packet.create(nativeCreateInt32(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createInt64(long value) {
    return Packet.create(nativeCreateInt64(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createFloat32(float value) {
    return Packet.create(nativeCreateFloat32(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createFloat64(double value) {
    return Packet.create(nativeCreateFloat64(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createBool(boolean value) {
    return Packet.create(nativeCreateBool(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createString(String value) {
    return Packet.create(nativeCreateString(mediapipeGraph.getNativeHandle(), value));
  }

  public Packet createInt16Vector(short[] data) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public Packet createInt32Vector(int[] data) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public Packet createInt64Vector(long[] data) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public Packet createFloat32Vector(float[] data) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public Packet createFloat64Vector(double[] data) {
    throw new UnsupportedOperationException("Not implemented yet");
  }

  public Packet createInt32Array(int[] data) {
    return Packet.create(nativeCreateInt32Array(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createFloat32Array(float[] data) {
    return Packet.create(nativeCreateFloat32Array(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createByteArray(byte[] data) {
    return Packet.create(nativeCreateStringFromByteArray(mediapipeGraph.getNativeHandle(), data));
  }

  /**
   * Creates a VideoHeader to be used by the calculator that requires it.
   *
   * <p>Note: we are not populating frame rate and duration. If the calculator needs those values,
   * the calculator is not suitable here. Modify the calculator to not require those values to work.
   */
  public Packet createVideoHeader(int width, int height) {
    return Packet.create(nativeCreateVideoHeader(mediapipeGraph.getNativeHandle(), width, height));
  }

  /**
   * Creates a mediapipe::TimeSeriesHeader, which is used by many audio related calculators.
   *
   * @param numChannels number of audio channels.
   * @param sampleRate sampling rate in Hertz.
   */
  public Packet createTimeSeriesHeader(int numChannels, double sampleRate) {
    return Packet.create(
        nativeCreateTimeSeriesHeader(mediapipeGraph.getNativeHandle(), numChannels, sampleRate));
  }

  public Packet createMatrix(int rows, int cols, float[] data) {
    return Packet.create(nativeCreateMatrix(mediapipeGraph.getNativeHandle(), rows, cols, data));
  }

  /** Creates a {@link Packet} containing the serialized proto string. */
  public Packet createSerializedProto(MessageLite message) {
    return Packet.create(
        nativeCreateStringFromByteArray(mediapipeGraph.getNativeHandle(), message.toByteArray()));
  }

  /** Creates a {@link Packet} containing a {@code CalculatorOptions} proto message. */
  public Packet createCalculatorOptions(MessageLite message) {
    return Packet.create(
        nativeCreateCalculatorOptions(mediapipeGraph.getNativeHandle(), message.toByteArray()));
  }

  /** Creates a {@link Packet} containing a protobuf MessageLite. */
  public Packet createProto(MessageLite message) {
    SerializedMessage serialized = PacketUtil.pack(message);
    return Packet.create(
        nativeCreateProto(mediapipeGraph.getNativeHandle(), serialized));
  }

  /** Creates a {@link Packet} containing the given camera intrinsics. */
  public Packet createCameraIntrinsics(
      float fx, float fy, float cx, float cy, float width, float height) {
    return Packet.create(
        nativeCreateCameraIntrinsics(
            mediapipeGraph.getNativeHandle(), fx, fy, cx, cy, width, height));
  }

  /**
   * Creates a mediapipe::GpuBuffer with the specified texture name and dimensions.
   *
   * @param name the OpenGL texture name.
   * @param width the width in pixels.
   * @param height the height in pixels.
   * @param releaseCallback a callback to be invoked when the mediapipe::GpuBuffer is released. Can be
   *     null.
   */
  public Packet createGpuBuffer(
      int name, int width, int height, TextureReleaseCallback releaseCallback) {
    return Packet.create(
        nativeCreateGpuBuffer(
            mediapipeGraph.getNativeHandle(), name, width, height, releaseCallback));
  }

  /**
   * Creates a mediapipe::GpuBuffer with the specified texture name and dimensions.
   *
   * @param name the OpenGL texture name.
   * @param width the width in pixels.
   * @param height the height in pixels.
   * @deprecated use {@link #createGpuBuffer(int,int,int,TextureReleaseCallback)} instead.
   */
  @Deprecated
  public Packet createGpuBuffer(int name, int width, int height) {
    return Packet.create(
        nativeCreateGpuBuffer(mediapipeGraph.getNativeHandle(), name, width, height, null));
  }

  /**
   * Creates a mediapipe::GpuBuffer with the provided {@link TextureFrame}.
   *
   * <p>Note: in order for MediaPipe to be able to access the texture, the application's GL context
   * must be linked with MediaPipe's. This is ensured by calling {@link
   * Graph#createGlRunner(String,long)} with the native handle to the application's GL context as
   * the second argument.
   */
  public Packet createGpuBuffer(TextureFrame frame) {
    return Packet.create(
        nativeCreateGpuBuffer(
            mediapipeGraph.getNativeHandle(),
            frame.getTextureName(),
            frame.getWidth(),
            frame.getHeight(),
            frame));
  }

  /** Helper callback adaptor to create the Java {@link GlSyncToken}. This is called by JNI code. */
  private void releaseWithSyncToken(long nativeSyncToken, TextureReleaseCallback releaseCallback) {
    releaseCallback.release(new GraphGlSyncToken(nativeSyncToken));
  }

  private native long nativeCreateReferencePacket(long context, long packet);
  private native long nativeCreateRgbImage(long context, ByteBuffer buffer, int width, int height);

  private native long nativeCreateAudioPacket(
      long context, byte[] data, int offset, int numChannels, int numSamples);

  private native long nativeCreateAudioPacketDirect(
      long context, ByteBuffer data, int numChannels, int numSamples);

  private native long nativeCreateRgbImageFromRgba(
      long context, ByteBuffer buffer, int width, int height);

  private native long nativeCreateGrayscaleImage(
      long context, ByteBuffer buffer, int width, int height);

  private native long nativeCreateRgbaImageFrame(
      long context, ByteBuffer buffer, int width, int height);
  private native long nativeCreateFloatImageFrame(
      long context, FloatBuffer buffer, int width, int height);
  private native long nativeCreateInt16(long context, short value);
  private native long nativeCreateInt32(long context, int value);
  private native long nativeCreateInt64(long context, long value);
  private native long nativeCreateFloat32(long context, float value);
  private native long nativeCreateFloat64(long context, double value);
  private native long nativeCreateBool(long context, boolean value);
  private native long nativeCreateString(long context, String value);
  private native long nativeCreateVideoHeader(long context, int width, int height);
  private native long nativeCreateTimeSeriesHeader(
      long context, int numChannels, double sampleRate);
  private native long nativeCreateMatrix(long context, int rows, int cols, float[] data);
  private native long nativeCreateGpuBuffer(
      long context, int name, int width, int height, TextureReleaseCallback releaseCallback);
  private native long nativeCreateInt32Array(long context, int[] data);
  private native long nativeCreateFloat32Array(long context, float[] data);
  private native long nativeCreateStringFromByteArray(long context, byte[] data);
  private native long nativeCreateProto(long context, SerializedMessage data);

  private native long nativeCreateCalculatorOptions(long context, byte[] data);

  private native long nativeCreateCameraIntrinsics(
      long context, float fx, float fy, float cx, float cy, float width, float height);
}
