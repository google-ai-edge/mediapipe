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

import android.graphics.Bitmap;
import com.google.mediapipe.framework.AndroidPacketGetter;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.TextureFrame;
import java.util.List;

/**
 * The base class of any MediaPipe image solution result. The base class contains the common parts
 * across all image solution results, including the input timestamp and the input image data. A new
 * MediaPipe image solution result class should extend ImageSolutionResult.
 */
public class ImageSolutionResult implements SolutionResult {
  protected long timestamp;
  protected Packet imagePacket;
  private Bitmap cachedBitmap;
  // A list of the output image packets produced by the graph.
  protected List<Packet> imageResultPackets;

  // Result timestamp, which is set to the timestamp of the corresponding input image. May return
  // Long.MIN_VALUE if the input image is not associated with a timestamp.
  @Override
  public long timestamp() {
    return timestamp;
  }

  // Returns the corresponding input image as a {@link Bitmap}.
  public Bitmap inputBitmap() {
    if (imagePacket == null) {
      return null;
    }
    if (cachedBitmap != null) {
      return cachedBitmap;
    }
    cachedBitmap = AndroidPacketGetter.getBitmapFromRgba(imagePacket);
    return cachedBitmap;
  }

  // Returns the corresponding input image as a {@link TextureFrame}. The caller must release the
  // acquired {@link TextureFrame} after using.
  public TextureFrame acquireInputTextureFrame() {
    if (imagePacket == null) {
      return null;
    }
    return PacketGetter.getTextureFrame(imagePacket);
  }

  // Releases image packet and the underlying data.
  void releaseImagePacket() {
    imagePacket.release();
    if (imageResultPackets != null) {
      for (Packet p : imageResultPackets) {
        p.release();
      }
    }
  }
}
