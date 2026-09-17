// Copyright 2026 The MediaPipe Authors.
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

package com.google.mediapipe.tasks.vision.overlay;

import android.graphics.Matrix;
import android.graphics.RectF;
import com.google.mediapipe.tasks.vision.core.RunningMode;

/**
 * Maps MediaPipe vision coordinates onto a view (Compose {@code Canvas}, {@link
 * android.view.View}, or bitmap).
 *
 * <p>Normalized landmarks ({@code x}, {@code y} in {@code [0, 1]}) are multiplied by the image
 * size, then scaled to the view. Bounding boxes from object / face detection are already in image
 * pixels and only need scale, optional camera rotation, and optional mirroring.
 *
 * <p><b>Running mode</b> matches CameraX / sample {@code OverlayView} behavior:
 *
 * <ul>
 *   <li>{@link RunningMode#IMAGE} and {@link RunningMode#VIDEO}: FIT_START — {@code min} scale so
 *       the whole image is visible (letterboxed).
 *   <li>{@link RunningMode#LIVE_STREAM}: FILL_START — {@code max} scale so the preview fills the
 *       view (cropped), matching {@code PreviewView.ScaleType.FILL_START}.
 * </ul>
 *
 * <p>Pass {@code mirrored = true} when the preview is a front camera that the UI flips
 * horizontally (typical Compose + CameraX selfie). Landmark {@code x} becomes {@code 1 - x}.
 */
public final class OverlayLayout {
  private final int viewWidth;
  private final int viewHeight;
  private final int imageWidth;
  private final int imageHeight;
  private final int rotationDegrees;
  private final boolean mirrored;
  private final float scale;
  private final int rotatedImageWidth;
  private final int rotatedImageHeight;

  private OverlayLayout(
      int viewWidth,
      int viewHeight,
      int imageWidth,
      int imageHeight,
      int rotationDegrees,
      boolean mirrored,
      float scale,
      int rotatedImageWidth,
      int rotatedImageHeight) {
    this.viewWidth = viewWidth;
    this.viewHeight = viewHeight;
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.rotationDegrees = rotationDegrees;
    this.mirrored = mirrored;
    this.scale = scale;
    this.rotatedImageWidth = rotatedImageWidth;
    this.rotatedImageHeight = rotatedImageHeight;
  }

  /**
   * Builds a layout for a view that displays {@code imageWidth} x {@code imageHeight} results.
   *
   * @param viewWidth overlay width in pixels (Compose {@code size.width}, or {@code View} content
   *     width)
   * @param viewHeight overlay height in pixels
   * @param imageWidth width of the image the task ran on, in pixels
   * @param imageHeight height of the image the task ran on, in pixels
   * @param runningMode IMAGE/VIDEO letterbox; LIVE_STREAM crop-to-fill
   */
  public static OverlayLayout create(
      int viewWidth, int viewHeight, int imageWidth, int imageHeight, RunningMode runningMode) {
    return create(
        viewWidth,
        viewHeight,
        imageWidth,
        imageHeight,
        runningMode,
        /* rotationDegrees= */ 0,
        /* mirrored= */ false);
  }

  /**
   * Builds a layout with optional camera-frame rotation and front-camera mirroring.
   *
   * @param rotationDegrees clockwise rotation of the <em>camera frame</em> relative to the overlay
   *     (0/90/180/270). Used for object/face <em>bounding boxes</em> that are in sensor
   *     orientation. Normalized landmarks from a task that already applied {@code
   *     ImageProcessingOptions.rotationDegrees} should pass {@code 0} and the rotated frame size.
   * @param mirrored {@code true} to flip landmark {@code x} and boxes horizontally (front camera)
   */
  public static OverlayLayout create(
      int viewWidth,
      int viewHeight,
      int imageWidth,
      int imageHeight,
      RunningMode runningMode,
      int rotationDegrees,
      boolean mirrored) {
    if (viewWidth <= 0 || viewHeight <= 0) {
      throw new IllegalArgumentException(
          "View size must be positive, found: " + viewWidth + "x" + viewHeight);
    }
    if (imageWidth <= 0 || imageHeight <= 0) {
      throw new IllegalArgumentException(
          "Image size must be positive, found: " + imageWidth + "x" + imageHeight);
    }
    int rotation = normalizeRotation(rotationDegrees);
    int rotatedWidth = isSwapped(rotation) ? imageHeight : imageWidth;
    int rotatedHeight = isSwapped(rotation) ? imageWidth : imageHeight;
    float widthRatio = viewWidth / (float) rotatedWidth;
    float heightRatio = viewHeight / (float) rotatedHeight;
    float scale =
        runningMode == RunningMode.LIVE_STREAM
            ? Math.max(widthRatio, heightRatio)
            : Math.min(widthRatio, heightRatio);
    return new OverlayLayout(
        viewWidth,
        viewHeight,
        imageWidth,
        imageHeight,
        rotation,
        mirrored,
        scale,
        rotatedWidth,
        rotatedHeight);
  }

  /** Overlay width in pixels. */
  public int viewWidth() {
    return viewWidth;
  }

  /** Overlay height in pixels. */
  public int viewHeight() {
    return viewHeight;
  }

  /** Unrotated image width in pixels. */
  public int imageWidth() {
    return imageWidth;
  }

  /** Unrotated image height in pixels. */
  public int imageHeight() {
    return imageHeight;
  }

  /** Normalized clockwise rotation in {@code {0, 90, 180, 270}}. */
  public int rotationDegrees() {
    return rotationDegrees;
  }

  /** Whether normalized {@code x} and boxes are flipped for a front camera. */
  public boolean mirrored() {
    return mirrored;
  }

  /**
   * Scale applied after rotation. IMAGE/VIDEO: {@code min} (fit). LIVE_STREAM: {@code max}
   * (fill).
   */
  public float scale() {
    return scale;
  }

  /** Image width after applying {@link #rotationDegrees()}. */
  public int rotatedImageWidth() {
    return rotatedImageWidth;
  }

  /** Image height after applying {@link #rotationDegrees()}. */
  public int rotatedImageHeight() {
    return rotatedImageHeight;
  }

  /** Maps a normalized landmark X in {@code [0, 1]} to overlay pixels. */
  public float mapX(float normalizedX) {
    float x = mirrored ? 1.f - normalizedX : normalizedX;
    return x * rotatedImageWidth * scale;
  }

  /** Maps a normalized landmark Y in {@code [0, 1]} to overlay pixels. */
  public float mapY(float normalizedY) {
    return normalizedY * rotatedImageHeight * scale;
  }

  /**
   * Maps a bounding box in <em>unrotated image pixels</em> to overlay pixels, applying rotation
   * then scale (and mirroring).
   */
  public RectF mapImageRect(RectF boxInImagePixels) {
    if (boxInImagePixels == null) {
      throw new IllegalArgumentException("Bounding box must not be null");
    }
    RectF mapped = new RectF(boxInImagePixels);
    if (rotationDegrees != 0) {
      Matrix matrix = new Matrix();
      matrix.postTranslate(-imageWidth / 2.f, -imageHeight / 2.f);
      matrix.postRotate(rotationDegrees);
      if (isSwapped(rotationDegrees)) {
        matrix.postTranslate(imageHeight / 2.f, imageWidth / 2.f);
      } else {
        matrix.postTranslate(imageWidth / 2.f, imageHeight / 2.f);
      }
      matrix.mapRect(mapped);
    }
    if (mirrored) {
      float left = rotatedImageWidth - mapped.right;
      float right = rotatedImageWidth - mapped.left;
      mapped.left = left;
      mapped.right = right;
    }
    mapped.left *= scale;
    mapped.top *= scale;
    mapped.right *= scale;
    mapped.bottom *= scale;
    return mapped;
  }

  static int normalizeRotation(int rotationDegrees) {
    int rotation = rotationDegrees % 360;
    if (rotation < 0) {
      rotation += 360;
    }
    if (rotation % 90 != 0) {
      throw new IllegalArgumentException(
          "Expected rotation to be a multiple of 90°, found: " + rotationDegrees);
    }
    return rotation;
  }

  private static boolean isSwapped(int rotationDegrees) {
    return rotationDegrees == 90 || rotationDegrees == 270;
  }
}
