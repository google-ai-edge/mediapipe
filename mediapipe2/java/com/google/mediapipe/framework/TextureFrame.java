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

/**
 * Interface for a video frame that can be accessed as a texture.
 *
 * <p>This interface defines a producer/consumer relationship between the component that originates
 * the TextureFrame and the component that receives it. The consumer <b>must</b> call {@link
 * #release()} when it is done using the frame. This gives the producer the opportunity to recycle
 * the resource.
 *
 * <p>When your application sends a TextureFrame into a MediaPipe graph, the application is the
 * producer and MediaPipe is the consumer. MediaPipe will call the release() method when all copies
 * of the packet holding the texture have been destroyed.
 *
 * <p>When MediaPipe sends a TextureFrame to the application, MediaPipe is the producer and the
 * application is the consumer. The application should call the release() method.
 *
 * <p>You can also send a TextureFrame from a component of your application to another. In this
 * case, the receiving component is the consumer, and should call release(). This can be useful, for
 * instance, if your application requires a "raw" mode where frames are sent directly from the video
 * source to the renderer, bypassing MediaPipe.
 */
public interface TextureFrame extends TextureReleaseCallback {
  /** The OpenGL name of the texture. */
  int getTextureName();

  /** Width of the frame in pixels. */
  int getWidth();

  /** Height of the frame in pixels. */
  int getHeight();

  /** The presentation time of the frame in microseconds **/
  long getTimestamp();

  /**
   * The consumer that receives this TextureFrame must call this method to inform the provider that
   * it is done with it.
   */
  void release();

  /**
   * If this texture is provided to MediaPipe, this method will be called when it is released. The
   * {@link GlSyncToken} can be used to wait for the GPU to be entirely done reading the texture.
   */
  @Override
  void release(GlSyncToken syncToken);
}
