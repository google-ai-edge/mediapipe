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
import java.util.ArrayList;
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
  // The cached texture frames.
  protected List<TextureFrame> imageResultTextureFrames;
  private TextureFrame cachedTextureFrame;

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

  // Returns the cached input image as a {@link TextureFrame}.
  public TextureFrame getCachedInputTextureFrame() {
    return cachedTextureFrame;
  }

  // Produces all texture frames from image packets and caches them for further use. The caller must
  // release the cached {@link TextureFrame}s after using.
  void produceAllTextureFrames() {
    cachedTextureFrame = acquireInputTextureFrame();
    if (imageResultPackets == null) {
      return;
    }
    imageResultTextureFrames = new ArrayList<>();
    for (Packet p : imageResultPackets) {
      imageResultTextureFrames.add(PacketGetter.getTextureFrame(p));
    }
  }

  // Releases all cached {@link TextureFrame}s.
  void releaseCachedTextureFrames() {
    if (cachedTextureFrame != null) {
      cachedTextureFrame.release();
    }
    if (imageResultTextureFrames != null) {
      for (TextureFrame textureFrame : imageResultTextureFrames) {
        textureFrame.release();
      }
    }
  }

  // Clears the underlying image packets to prevent the callers from accessing the invalid packets
  // outside of the output callback method.
  void clearImagePackets() {
    imagePacket = null;
    if (imageResultPackets != null) {
      imageResultPackets.clear();
    }
  }
}
