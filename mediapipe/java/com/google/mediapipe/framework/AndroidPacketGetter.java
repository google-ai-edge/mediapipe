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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

/**
 * Android-specific subclass of PacketGetter.
 *
 * <p>See {@link PacketGetter} for general information.
 *
 * <p>This class contains methods that are Android-specific.
 */
public final class AndroidPacketGetter {

  /**
   * Gets a {@link Bitmap} from a mediapipe image packet, using the number of channels of the input
   * image.
   *
   * @param packet mediapipe packet a RGB, RGBA or Gray8 image
   * @throws UnsupportedOperationException if the packet does not contain an image in the supported
   *     formats.
   * @return {@link Bitmap} with pixels copied from the packet
   */
  public static Bitmap getBitmap(Packet packet) {
    // Note that instead of using the image format, we use the number of channels to determine the
    // format. This avoids the need to have a dependency on  `MpImage` in the core MediaPipe
    // framework and allows us to use the same logic we use in
    // `nativeCreateCpuImage`.
    int channels = PacketGetter.getImageNumChannels(packet);
    switch (channels) {
      case 1:
        return getBitmapFromGray8(packet);
      case 3:
        return getBitmapFromRgb(packet);
      case 4:
        return getBitmapFromRgba(packet);
      default:
        throw new UnsupportedOperationException("Unsupported number of channels: " + channels);
    }
  }

  /**
   * Gets an {@code ALPHA_8} bitmap from an 8-bit single channel mediapipe image frame packet.
   *
   * @param packet mediapipe packet
   * @return {@link Bitmap} with pixels copied from the packet
   */
  public static Bitmap getBitmapFromGray8(Packet packet) {
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    Bitmap bitmap = Bitmap.createBitmap(width, height, Config.ALPHA_8);
    copyGray8ToBitmap(packet, bitmap, width, height);
    return bitmap;
  }

  /**
   * Copies data from an 8-bit alpha mediapipe image frame packet to {@code ALPHA_8} bitmap.
   *
   * @param packet mediapipe packet
   * @param inBitmap mutable {@link Bitmap} of same dimension and config as the expected output, the
   *     image would be copied to this {@link Bitmap}
   */
  public static void copyGray8ToBitmap(Packet packet, Bitmap inBitmap) {
    checkArgument(inBitmap.isMutable(), "Input bitmap should be mutable.");
    checkArgument(
        inBitmap.getConfig() == Config.ALPHA_8, "Input bitmap should be of type ALPHA_8.");
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    checkArgument(inBitmap.getByteCount() == width * height, "Input bitmap size mismatch.");
    copyGray8ToBitmap(packet, inBitmap, width, height);
  }

  private static void copyGray8ToBitmap(
      Packet packet, Bitmap mutableBitmap, int width, int height) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height);
    boolean status = PacketGetter.getImageData(packet, buffer);
    checkState(
        status,
        String.format(
            Locale.getDefault(),
            "Got error from getImageData, returning null Bitmap. Image width %d, height %d",
            width,
            height));
    mutableBitmap.copyPixelsFromBuffer(buffer);
  }

  /**
   * Gets an {@code ARGB_8888} bitmap from an RGB mediapipe image frame packet.
   *
   * @param packet mediapipe packet
   * @return {@link Bitmap} with pixels copied from the packet
   */
  public static Bitmap getBitmapFromRgb(Packet packet) {
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    Bitmap bitmap = Bitmap.createBitmap(width, height, Config.ARGB_8888);
    copyRgbToBitmap(packet, bitmap, width, height);
    return bitmap;
  }

  /**
   * Copies data from an RGB mediapipe image frame packet to {@code ARGB_8888} bitmap.
   *
   * @param packet mediapipe packet
   * @param inBitmap mutable {@link Bitmap} of same dimension and config as the expected output, the
   *     image would be copied to this {@link Bitmap}
   */
  public static void copyRgbToBitmap(Packet packet, Bitmap inBitmap) {
    checkArgument(inBitmap.isMutable(), "Input bitmap should be mutable.");
    checkArgument(
        inBitmap.getConfig() == Config.ARGB_8888, "Input bitmap should be of type ARGB_8888.");
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    checkArgument(inBitmap.getByteCount() == width * height * 4, "Input bitmap size mismatch.");
    copyRgbToBitmap(packet, inBitmap, width, height);
  }

  private static void copyRgbToBitmap(Packet packet, Bitmap mutableBitmap, int width, int height) {
    // TODO: use NDK Bitmap access instead of copyPixelsToBuffer.
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
    PacketGetter.getRgbaFromRgb(packet, buffer);
    mutableBitmap.copyPixelsFromBuffer(buffer);
  }

  /**
   * Gets an {@code ARGB_8888} bitmap from an RGBA mediapipe image frame packet. Returns null in
   * case of failure.
   *
   * @param packet mediapipe packet
   * @return {@link Bitmap} with pixels copied from the packet
   */
  public static Bitmap getBitmapFromRgba(Packet packet) {
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    Bitmap bitmap = Bitmap.createBitmap(width, height, Config.ARGB_8888);
    copyRgbaToBitmap(packet, bitmap, width, height);
    return bitmap;
  }

  /**
   * Copies data from an RGBA mediapipe image frame packet to {@code ARGB_8888} bitmap.
   *
   * @param packet mediapipe packet
   * @param inBitmap mutable {@link Bitmap} of same dimension and config as the expected output, the
   *     image would be copied to this {@link Bitmap}.
   */
  public static void copyRgbaToBitmap(Packet packet, Bitmap inBitmap) {
    checkArgument(inBitmap.isMutable(), "Input bitmap should be mutable.");
    checkArgument(
        inBitmap.getConfig() == Config.ARGB_8888, "Input bitmap should be of type ARGB_8888.");
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    checkArgument(inBitmap.getByteCount() == width * height * 4, "Input bitmap size mismatch.");
    copyRgbaToBitmap(packet, inBitmap, width, height);
  }

  private static void copyRgbaToBitmap(Packet packet, Bitmap mutableBitmap, int width, int height) {
    // TODO: unify into a single getBitmap call.
    // TODO: use NDK Bitmap access instead of copyPixelsToBuffer.
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
    buffer.order(ByteOrder.nativeOrder());
    // Note: even though the Android Bitmap config is named ARGB_8888, the data
    // is stored as RGBA internally.
    boolean status = PacketGetter.getImageData(packet, buffer);
    checkState(
        status,
        String.format(
            Locale.getDefault(),
            "Got error from getImageData, returning null Bitmap. Image width %d, height %d",
            width,
            height));
    mutableBitmap.copyPixelsFromBuffer(buffer);
  }

  private AndroidPacketGetter() {}
}
