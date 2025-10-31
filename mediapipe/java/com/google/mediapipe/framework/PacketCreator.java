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

import com.google.mediapipe.framework.ProtoUtil.SerializedMessage;
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

  /** Sync mode to use when creating a GpuBuffer packet. */
  public static enum SyncMode {
    // External texture is already up-to-date and can be used on a shared context
    // as is (e.g. prior glFinish call) or there's just a single GL context used
    // for both external textures and MediaPipe graph.
    NO_SYNC(0),
    // MediaPipe graph has a dedicated GL context(s) and external texture must be
    // efficiently synchronized using GL sync object.
    SYNC(1),
    // MediaPipe graph has a dedicated GL context(s) and external texture can be
    // synchronized using GL sync object or glFinish or can be skipped
    // alltogether.
    MAYBE_SYNC_OR_FINISH(2);

    private final int value;

    SyncMode(int value) {
      this.value = value;
    }
  }

  // Locally declared GL constant so that we do not have any platform dependencies.
  private static final int GL_RGBA = 0x1908;
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
      throw new IllegalArgumentException(
          "The size of the buffer should be: "
              + widthStep * height
              + " but is "
              + buffer.capacity());
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
   * @param data the raw audio data, bytes per sample is 2(only AudioFormat.ENCODING_PCM_16BIT is
   *     supported). Must either be a direct byte buffer or have an array, and the data has to be
   *     FILLED with ByteOrder.LITTLE_ENDIAN byte order, which is ByteOrder.nativeOrder() on Android
   *     (https://developer.android.com/ndk/guides/abis.html).
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
      throw new IllegalArgumentException(
          "The size of the buffer should be: "
              + width * height * 4
              + " but is "
              + buffer.capacity());
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
      throw new IllegalArgumentException(
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
      throw new IllegalArgumentException(
          "The size of the buffer should be: "
              + width * height * 4
              + " but is "
              + buffer.capacity());
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
      throw new IllegalArgumentException(
          "The size of the buffer should be: "
              + width * height * 4
              + " but is "
              + buffer.capacity());
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
    return Packet.create(nativeCreateInt16Vector(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createInt32Vector(int[] data) {
    return Packet.create(nativeCreateInt32Vector(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createInt64Vector(long[] data) {
    return Packet.create(nativeCreateInt64Vector(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createFloat32Vector(float[] data) {
    return Packet.create(nativeCreateFloat32Vector(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createFloat64Vector(double[] data) {
    return Packet.create(nativeCreateFloat64Vector(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createInt32Array(int[] data) {
    return Packet.create(nativeCreateInt32Array(mediapipeGraph.getNativeHandle(), data));
  }

  public Packet createInt32Pair(int first, int second) {
    return Packet.create(nativeCreateInt32Pair(mediapipeGraph.getNativeHandle(), first, second));
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
    SerializedMessage serialized = ProtoUtil.pack(message);
    return Packet.create(nativeCreateProto(mediapipeGraph.getNativeHandle(), serialized));
  }

  /** Creates a {@link Packet} containing the given camera intrinsics. */
  public Packet createCameraIntrinsics(
      float fx, float fy, float cx, float cy, float width, float height) {
    return Packet.create(
        nativeCreateCameraIntrinsics(
            mediapipeGraph.getNativeHandle(), fx, fy, cx, cy, width, height));
  }

  /**
   * Creates a mediapipe::GpuBuffer with the specified texture name, dimensions, and format.
   *
   * @param name the OpenGL texture name.
   * @param width the width in pixels.
   * @param height the height in pixels.
   * @param format the OpenGL texture format.
   * @param releaseCallback a callback to be invoked when the mediapipe::GpuBuffer is released. Can
   *     be null.
   * @param syncMode a sync mode required for proper synchronization between external and MediaPipe
   *     contexts.
   */
  public Packet createGpuBuffer(
      int name,
      int width,
      int height,
      int format,
      TextureReleaseCallback releaseCallback,
      SyncMode syncMode) {
    return Packet.create(
        nativeCreateGpuBuffer(
            mediapipeGraph.getNativeHandle(),
            name,
            width,
            height,
            format,
            releaseCallback,
            syncMode.value));
  }

  /**
   * Same as above, but doesn't request any synchronization. NOTE: prefer methods with explicit sync
   * mode to avoid unexpected synchronization issues.
   */
  public Packet createGpuBuffer(
      int name, int width, int height, int format, TextureReleaseCallback releaseCallback) {
    return createGpuBuffer(name, width, height, format, releaseCallback, SyncMode.NO_SYNC);
  }

  /**
   * Creates a mediapipe::GpuBuffer with the specified texture name and dimensions.
   *
   * @param name the OpenGL texture name.
   * @param width the width in pixels.
   * @param height the height in pixels.
   * @param releaseCallback a callback to be invoked when the mediapipe::GpuBuffer is released. Can
   *     be null.
   * @param syncMode a sync mode required for proper synchronization between external and MediaPipe
   *     contexts.
   */
  public Packet createGpuBuffer(
      int name, int width, int height, TextureReleaseCallback releaseCallback, SyncMode syncMode) {
    return createGpuBuffer(name, width, height, GL_RGBA, releaseCallback, syncMode);
  }

  /**
   * Same as above, but doesn't request any synchronization. NOTE: prefer methods with explicit sync
   * mode to avoid unexpected synchronization issues.
   */
  public Packet createGpuBuffer(
      int name, int width, int height, TextureReleaseCallback releaseCallback) {
    return createGpuBuffer(name, width, height, releaseCallback, SyncMode.NO_SYNC);
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
    return createGpuBuffer(name, width, height, GL_RGBA, null);
  }

  /**
   * Creates a mediapipe::GpuBuffer with the provided {@link TextureFrame}.
   *
   * <p>Note: in order for MediaPipe to be able to access the texture, the application's GL context
   * must be linked with MediaPipe's. This is ensured by calling {@link
   * Graph#createGlRunner(String,long)} with the native handle to the application's GL context as
   * the second argument.
   */
  public Packet createGpuBuffer(TextureFrame frame, SyncMode syncMode) {
    return Packet.create(
        nativeCreateGpuBuffer(
            mediapipeGraph.getNativeHandle(),
            frame.getTextureName(),
            frame.getWidth(),
            frame.getHeight(),
            frame.getFormat(),
            frame,
            syncMode.value));
  }

  /**
   * Same as above, but doesn't request any synchronization. NOTE: prefer methods with explicit sync
   * mode to avoid unexpected synchronization issues.
   */
  public Packet createGpuBuffer(TextureFrame frame) {
    return createGpuBuffer(frame, SyncMode.NO_SYNC);
  }

  /**
   * Creates a mediapipe::Image with the provided {@link TextureFrame}.
   *
   * <p>Note: in order for MediaPipe to be able to access the texture, the application's GL context
   * must be linked with MediaPipe's. This is ensured by calling {@link
   * Graph#createGlRunner(String,long)} with the native handle to the application's GL context as
   * the second argument.
   */
  public Packet createImage(TextureFrame frame) {
    return Packet.create(
        nativeCreateGpuImage(
            mediapipeGraph.getNativeHandle(),
            frame.getTextureName(),
            frame.getWidth(),
            frame.getHeight(),
            frame.getFormat(),
            frame));
  }

  /**
   * Creates a 1, 3, or 4 channel 8-bit Image packet from a U8, RGB, or RGBA byte buffer.
   *
   * <p>Use {@link ByteBuffer#allocateDirect} when allocating the buffer.
   *
   * <p>For 3 and 4 channel images, the pixel rows should have 4-byte alignment.
   */
  public Packet createImage(ByteBuffer buffer, int width, int height, int numChannels) {
    int widthStep;
    if (numChannels == 4) {
      widthStep = width * 4;
    } else if (numChannels == 3) {
      widthStep = (((width * 3) + 3) / 4) * 4;
    } else if (numChannels == 1) {
      widthStep = width;
    } else {
      throw new IllegalArgumentException("Channels should be: 1, 3, or 4, but is " + numChannels);
    }
    int expectedSize = widthStep * height;
    if (buffer.capacity() != expectedSize) {
      throw new IllegalArgumentException(
          "The size of the buffer should be: " + expectedSize + " but is " + buffer.capacity());
    }
    return Packet.create(
        nativeCreateCpuImage(
            mediapipeGraph.getNativeHandle(), buffer, width, height, widthStep, numChannels));
  }

  /** Helper callback adaptor to create the Java {@link GlSyncToken}. This is called by JNI code. */
  private void releaseWithSyncToken(long nativeSyncToken, TextureReleaseCallback releaseCallback) {
    releaseCallback.release(new GraphGlSyncToken(nativeSyncToken));
  }

  private native long nativeCreateReferencePacket(long context, long packet);

  private native long nativeCreateAudioPacket(
      long context, byte[] data, int offset, int numChannels, int numSamples);

  private native long nativeCreateAudioPacketDirect(
      long context, ByteBuffer data, int numChannels, int numSamples);

  private native long nativeCreateRgbImageFromRgba(
      long context, ByteBuffer buffer, int width, int height);

  private native long nativeCreateRgbImage(long context, ByteBuffer buffer, int width, int height);

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
      long context,
      int name,
      int width,
      int height,
      int format,
      TextureReleaseCallback releaseCallback,
      int sync);

  private native long nativeCreateGpuImage(
      long context, int name, int width, int height, int format,
      TextureReleaseCallback releaseCallback);

  private native long nativeCreateCpuImage(
      long context, ByteBuffer buffer, int width, int height, int rowBytes, int numChannels);

  private native long nativeCreateInt32Array(long context, int[] data);

  private native long nativeCreateInt32Pair(long context, int first, int second);

  private native long nativeCreateFloat32Array(long context, float[] data);

  private native long nativeCreateInt16Vector(long context, short[] data);
  private native long nativeCreateInt32Vector(long context, int[] data);
  private native long nativeCreateInt64Vector(long context, long[] data);
  private native long nativeCreateFloat32Vector(long context, float[] data);
  private native long nativeCreateFloat64Vector(long context, double[] data);

  private native long nativeCreateStringFromByteArray(long context, byte[] data);

  private native long nativeCreateProto(long context, SerializedMessage data);

  private native long nativeCreateCalculatorOptions(long context, byte[] data);

  private native long nativeCreateCameraIntrinsics(
      long context, float fx, float fy, float cx, float cy, float width, float height);
}
