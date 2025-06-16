// Copyright 2023 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.genai.llminference;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.media.Image;
import com.google.mediapipe.framework.image.BitmapExtractor;
import com.google.mediapipe.framework.image.ByteBufferExtractor;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.framework.image.MPImageProperties;
import com.google.mediapipe.framework.image.MediaImageExtractor;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmOptionsProto.LlmModelSettings;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmOptionsProto.LlmSessionConfig;
import com.google.mediapipe.tasks.genai.llminference.jni.proto.LlmResponseContextProto.LlmResponseContext;
import com.google.protobuf.InvalidProtocolBufferException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Internal Task Runner class for all LLM Tasks.
 *
 * @hide
 */
final class LlmTaskRunner implements AutoCloseable {
  private final long engineHandle;
  private final AtomicBoolean isLocked = new AtomicBoolean(false);

  private ProgressListener<List<String>> resultListener = (unused1, unused2) -> {};

  /**
   * Describes how pixel bits encode color. A pixel may be an alpha mask, a grayscale, RGB, or ARGB.
   *
   * <p>This matches the SkColorType enum in https://api.skia.org/SkColorType_8h.html.
   */
  private enum SkColorType {
    /** Uninitialized. */
    UNKNOWN(0),
    /** Pixel with alpha in 8-bit byte. */
    ALPHA_8(1),
    /** Pixel with 5 bits red, 6 bits green, 5 bits blue, in 16-bit word. */
    RGB_565(2),
    /** Pixel with 4 bits for alpha, red, green, blue; in 16-bit word. */
    ARGB_4444(3),
    /** Pixel with 8 bits for red, green, blue, alpha; in 32-bit word. */
    RGBA_8888(4),
    /** Pixel with 8 bits each for red, green, blue; in 32-bit word. */
    RGB_888X(5),
    /** Pixel with 8 bits for blue, green, red, alpha; in 32-bit word. */
    BGRA_8888(6),
    /** 10 bits for red, green, blue; 2 bits for alpha; in 32-bit word. */
    RGBA_1010102(7),
    /** 10 bits for blue, green, red; 2 bits for alpha; in 32-bit word. */
    BGRA_1010102(8),
    /** Pixel with 10 bits each for red, green, blue; in 32-bit word. */
    RGB_101010X(9),
    /** Pixel with 10 bits each for blue, green, red; in 32-bit word. */
    BGR101010X(10),
    /** Pixel with 10 bits each for blue, green, red; in 32-bit word, extended range. */
    BGR_101010X_XR(11),
    /** Pixel with 10 bits each for blue, green, red, alpha; in 64-bit word, extended range. */
    BGRA_10101010_XR(12),
    /**
     * Pixel with 10 used bits (most significant) followed by 6 unused bits for red, green, blue,
     * alpha; in 64-bit word.
     */
    RGBA_10X6(13),
    /** Pixel with grayscale level in 8-bit byte. */
    GRAY_8(14),
    /** Pixel with half floats in [0,1] for red, green, blue, alpha; in 64-bit word. */
    RGBA_F16NORM(15),
    /** Pixel with half floats for red, green, blue, alpha; in 64-bit word. */
    RGBA_F16(16),
    /** Pixel with half floats for red, green, blue; in 64-bit word. */
    RGB_F16F16F16X(17),
    /** Pixel using C float for red, green, blue, alpha; in 128-bit word. */
    RGBA_F32(18),
    /** Pixel with a uint8_t for red and green. */
    R8G8_UNORM(19),
    /** Pixel with a half float for alpha. */
    A16_FLOAT(20),
    /** Pixel with a half float for red and green. */
    R16G16_FLOAT(21),
    /** Pixel with a little endian uint16_t for alpha. */
    A16_UNORM(22),
    /** Pixel with a little endian uint16_t for red and green. */
    R16G16_UNORM(23),
    /** Pixel with a little endian uint16_t for red, green, blue and alpha. */
    R16G16B16A16_UNORM(24),
    /** Pixel with 8 bits for red, green, blue, alpha; in 32-bit word, gamma encoded. */
    SRGBA_8888(25),
    /** Pixel with a uint8_t for red. */
    R8_UNORM(26);

    private final int value;

    SkColorType(int value) {
      this.value = value;
    }

    /** Returns the integer value associated with this color type. */
    int getValue() {
      return value;
    }
  }

  /**
   * Describes how to interpret the alpha component of a pixel. A pixel may be opaque, or alpha,
   * describing multiple levels of transparency.
   *
   * <p>This matches the SkColorType enum in https://api.skia.org/SkAlphaType_8h.html.
   */
  private enum SkAlphaType {
    UNIITALIZED(0),
    /** Pixel is opaque */
    OPAQUE(1),
    /** Pixel components are premultiplied by alpha */
    PREMULTIPLIED(2),
    /** Pixel components are independent of alpha */
    UNPREMULTIPLIED(3);

    private final int value;

    SkAlphaType(int value) {
      this.value = value;
    }

    /** Returns the integer value associated with this alpha type. */
    int getValue() {
      return value;
    }
  };

  /** The session to use for LLM inference calls. */
  public static final class LlmSession {
    private final long sessionHandle;
    private LlmSessionConfig sessionConfig;

    LlmSession(long sessionHandle, LlmSessionConfig sessionConfig) {
      this.sessionHandle = sessionHandle;
      this.sessionConfig = sessionConfig;
    }

    public LlmSessionConfig getSessionConfig() {
      return sessionConfig;
    }
  }

  public LlmTaskRunner(Context context, String taskName, LlmModelSettings modelSettings) {
    this.engineHandle = nativeCreateEngine(modelSettings.toByteArray());
  }

  /** Creates a new LLM session. */
  public LlmSession createSession(LlmSessionConfig sessionConfig) {
    long sessionHandle = nativeCreateSession(sessionConfig.toByteArray(), engineHandle);
    return new LlmSession(sessionHandle, sessionConfig);
  }

  /** Adds a new query to the session context. */
  public void addQueryChunk(LlmSession session, String input) {
    nativeAddQueryChunk(session.sessionHandle, input);
  }

  /** Adds a new image to the session context. */
  public void addImage(LlmSession session, MPImage input) {
    long imageHandle = createImage(input);
    try {
      nativeAddImage(session.sessionHandle, imageHandle);
    } finally {
      nativeDeleteSkBitmap(imageHandle);
    }
  }

  /**
   * Adds a new audio to the session context.
   *
   * @param session The LlmSession to add the audio spectrum to.
   * @param rawAudioData A array of byte values representing the audio data.
   */
  public void addAudio(LlmSession session, byte[] rawAudioData) {
    if (rawAudioData == null) {
      throw new IllegalArgumentException("Audio data cannot be null.");
    }

    nativeAddAudio(engineHandle, session.sessionHandle, rawAudioData);
  }

  /**
   * Returns the SentencePieceProcessor associated with the LLM engine.
   *
   * <p>The returned SentencePieceProcessor is owned by the LLM engine and should not be deleted by
   * the caller.
   */
  final long getSentencePieceProcessor() {
    return nativeGetSentencePieceProcessor(engineHandle);
  }

  /** Invokes the LLM with the given session and waits for the result. */
  public List<String> predictSync(LlmSession session) {
      byte[] responseBytes = nativePredictSync(session.sessionHandle);
    return parseResponse(responseBytes).getResponsesList();
  }

  /** Invokes the LLM with the given session and calls the callback with the result. */
  public void predictAsync(LlmSession session, long callbackHandle) {
    nativePredictAsync(session.sessionHandle, callbackHandle);
  }

  /** Cancels pending processes in the session. */
  public void pendingProcessCancellation(LlmSession session) {
    nativePendingProcessCancellation(session.sessionHandle);
  }

  /** Invokes the native token cost calculator and returns the size of the string in tokens. */
  public int sizeInTokens(LlmSession session, String text) {
    return nativeSizeInTokens(session.sessionHandle, text);
  }

  /** Clones the current session. */
  public LlmSession cloneSession(LlmSession session) {
    long clonedSessionHandle = nativeCloneSession(session.sessionHandle);
    return new LlmSession(clonedSessionHandle, session.getSessionConfig());
  }

  /** Removes the session and frees up its context. */
  public void deleteSession(LlmSession session) {
    nativeDeleteSession(session.sessionHandle);
  }

  /** Updates the session config. */
  public void updateSessionConfig(LlmSession session, LlmSessionConfig config) {
    byte[] configBytes = config.toByteArray();
    session.sessionConfig = config;
    nativeUpdateSessionConfig(session.sessionHandle, configBytes);
  }

  long registerCallback(LlmTaskRunnerDelegate delegate) {
    return nativeRegisterCallback(delegate);
  }

  void unregisterCallback(long callbackHandle) {
    nativeRemoveCallback(callbackHandle);
  }

  LlmResponseContext parseResponse(byte[] response) {
    try {
      return LlmResponseContext.parseFrom(response);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalStateException("Failed to parse response", e);
    }
  }

  @Override
  public void close() {
    nativeDeleteEngine(engineHandle);
  }

  private long createImage(MPImage image) {
    MPImageProperties properties = image.getContainedImageProperties().get(0);

    SkAlphaType skAlphaType = SkAlphaType.OPAQUE;
    ByteBuffer buffer;
    SkColorType skColorType;

    int width = image.getWidth();
    int height = image.getHeight();

    if (properties.getStorageType() == MPImage.STORAGE_TYPE_BYTEBUFFER) {
      buffer = ByteBufferExtractor.extract(image);

      switch (properties.getImageFormat()) {
        case MPImage.IMAGE_FORMAT_RGBA:
          skColorType = SkColorType.RGBA_8888;
          break;
        case MPImage.IMAGE_FORMAT_RGB:
          skColorType = SkColorType.RGB_888X;
          break;
        case MPImage.IMAGE_FORMAT_ALPHA:
          skColorType = SkColorType.ALPHA_8;
          break;
        default:
          throw new UnsupportedOperationException(
              "Unsupported MediaPipe Image image format: " + properties.getImageFormat());
      }
    } else if (properties.getStorageType() == MPImage.STORAGE_TYPE_BITMAP) {
      Bitmap bitmap = BitmapExtractor.extract(image);
      if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
        throw new UnsupportedOperationException("Bitmap must use ARGB_8888 config.");
      }
      skColorType = SkColorType.RGBA_8888;

      buffer = ByteBuffer.allocateDirect(bitmap.getByteCount());
      bitmap.copyPixelsToBuffer(buffer);
    } else if (properties.getStorageType() == MPImage.STORAGE_TYPE_MEDIA_IMAGE) {
      Image mediaImage = MediaImageExtractor.extract(image);
      if (mediaImage.getFormat() != PixelFormat.RGBA_8888) {
        throw new UnsupportedOperationException("Android media image must use RGBA_8888 config.");
      }
      buffer = mediaImage.getPlanes()[0].getBuffer();
      skColorType = SkColorType.RGBA_8888;
    } else {
      throw new UnsupportedOperationException(
          "Unsupported Image container type: " + properties.getStorageType());
    }

    return nativeCreateSkBitmap(
        buffer, width, height, skColorType.getValue(), skAlphaType.getValue());
  }

  /**
   * Acquires a global LLM task lock.
   *
   * <p>This lock can be used to ensure that asynchronous operations are not run in parallel. Lock
   * manangement has to be done manually by the consuming APIs through this method, {@link
   * #releaseLock()} and {@link #isLocked()}.
   */
  void acquireLock() {
    // TODO: Move this lock to the session level.
    if (isLocked.getAndSet(true)) {
      throw new IllegalStateException("Cannot acquire LLM task lock while locked.");
    }
  }

  /**
   * Releases a global LLM task lock.
   *
   * <p>This lock can be used to ensure that asynchronous operations are not run in parallel. Lock
   * manangement has to be done manually by the consuming APIs through this method, {@link
   * #acquireLock()} and {@link #isLocked()}.
   */
  void releaseLock() {
    // TODO: Move this lock to the session level.
    if (!isLocked.getAndSet(false)) {
      throw new IllegalStateException("Cannot release LLM task lock while unlocked.");
    }
  }

  /**
   * Returns true if the global LLM task lock is held.
   *
   * <p>This lock can be used to ensure that asynchronous operations are not run in parallel. Lock
   * manangement has to be done manually by the consuming APIs through this method, {@link
   * #acquireLock()} and {@link #releaseLock()}.
   */
  boolean isLocked() {
    // TODO: Move this lock to the session level.
    return isLocked.get();
  }

  private static native long nativeCreateEngine(byte[] modelSettings);

  private static native void nativeDeleteEngine(long enginePointer);

  private static native long nativeCreateSession(byte[] sessionConfig, long enginePointer);

  private static native long nativeCloneSession(long sessionPointer);

  private static native void nativeDeleteSession(long sessionPointer);

  private static native void nativeAddQueryChunk(long sessionPointer, String input);

  private static native byte[] nativePredictSync(long sessionPointer);

  private static native long nativeRegisterCallback(Object callback);

  private static native void nativeRemoveCallback(long callbackHandle);

  private static native void nativePredictAsync(long sessionPointer, long callbackContextHandle);

  private static native void nativePendingProcessCancellation(long sessionPointer);

  private static native int nativeSizeInTokens(long sessionPointer, String input);

  private static native long nativeCreateSkBitmap(
      ByteBuffer buffer, int width, int height, int colorType, int alphaType);

  private static native void nativeDeleteSkBitmap(long imagePointer);

  private static native void nativeAddImage(long sessionPointer, long imagePointer);

  private static native long nativeGetSentencePieceProcessor(long enginePointer);

  private static native void nativeUpdateSessionConfig(long sessionPointer, byte[] config);

  private static native void nativeAddAudio(
      long enginePointer, long sessionPointer, byte[] rawAudioData);
}
