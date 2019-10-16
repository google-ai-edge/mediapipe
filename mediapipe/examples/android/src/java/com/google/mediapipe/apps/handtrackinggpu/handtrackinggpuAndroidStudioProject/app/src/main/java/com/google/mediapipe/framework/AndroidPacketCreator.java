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

// TODO: use Preconditions in this file.
/**
 * Android-specific subclass of PacketCreator.
 *
 * <p>See {@link PacketCreator} for general information.
 *
 * <p>This class contains methods that are Android-specific. You can (and should) use the base
 * PacketCreator on Android if you do not need any methods from this class.
 */
public class AndroidPacketCreator extends PacketCreator {
  public AndroidPacketCreator(Graph context) {
    super(context);
  }

  /** Creates a 3 channel RGB ImageFrame packet from a {@link Bitmap}. */
  public Packet createRgbImageFrame(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      throw new RuntimeException("bitmap must use ARGB_8888 config.");
    }
    return Packet.create(nativeCreateRgbImageFrame(mediapipeGraph.getNativeHandle(), bitmap));
  }

  /** Creates a 4 channel RGBA ImageFrame packet from a {@link Bitmap}. */
  public Packet createRgbaImageFrame(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      throw new RuntimeException("bitmap must use ARGB_8888 config.");
    }
    return Packet.create(nativeCreateRgbaImageFrame(mediapipeGraph.getNativeHandle(), bitmap));
  }

  /**
   * Returns the native handle of a new internal::PacketWithContext object on success. Returns 0 on
   * failure.
   */
  private native long nativeCreateRgbImageFrame(long context, Bitmap bitmap);

  /**
   * Returns the native handle of a new internal::PacketWithContext object on success. Returns 0 on
   * failure.
   */
  private native long nativeCreateRgbaImageFrame(long context, Bitmap bitmap);
}
