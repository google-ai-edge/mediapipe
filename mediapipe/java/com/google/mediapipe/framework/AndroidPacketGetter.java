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

import android.graphics.Bitmap;
import com.google.common.flogger.FluentLogger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Android-specific subclass of PacketGetter.
 *
 * <p>See {@link PacketGetter} for general information.
 *
 * <p>This class contains methods that are Android-specific.
 */
public final class AndroidPacketGetter {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  /** Gets an {@code ARGB_8888} bitmap from an RGB mediapipe image frame packet. */
  public static Bitmap getBitmapFromRgb(Packet packet) {
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
    PacketGetter.getRgbaFromRgb(packet, buffer);
    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    bitmap.copyPixelsFromBuffer(buffer);
    return bitmap;
  }

  /**
   * Gets an {@code ARGB_8888} bitmap from an RGBA mediapipe image frame packet. Returns null in
   * case of failure.
   */
  public static Bitmap getBitmapFromRgba(Packet packet) {
    // TODO: unify into a single getBitmap call.
    // TODO: use NDK Bitmap access instead of copyPixelsToBuffer.
    int width = PacketGetter.getImageWidth(packet);
    int height = PacketGetter.getImageHeight(packet);
    ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
    buffer.order(ByteOrder.nativeOrder());
    // Note: even though the Android Bitmap config is named ARGB_8888, the data
    // is stored as RGBA internally.
    boolean status = PacketGetter.getImageData(packet, buffer);
    if (!status) {
      logger.atSevere().log(
          "Got error from getImageData, returning null Bitmap. Image width %d, height %d",
          width, height);
      return null;
    }
    Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    bitmap.copyPixelsFromBuffer(buffer);
    return bitmap;
  }

  private AndroidPacketGetter() {}
}
