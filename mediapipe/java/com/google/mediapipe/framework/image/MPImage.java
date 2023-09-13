/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.mediapipe.framework.image;

import androidx.annotation.IntDef;
import androidx.annotation.Nullable;
import java.io.Closeable;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * The wrapper class for image objects.
 *
 * <p>{@link MPImage} is designed to be an immutable image container, which could be shared
 * cross-platforms.
 *
 * <p>To construct a {@link MPImage}, use the provided builders:
 *
 * <ul>
 *   <li>{@link ByteBufferImageBuilder}
 *   <li>{@link BitmapImageBuilder}
 *   <li>{@link MediaImageBuilder}
 * </ul>
 *
 * <p>{@link MPImage} uses reference counting to maintain internal storage. When it is created the
 * reference count is 1. Developer can call {@link #close()} to reduce reference count to release
 * internal storage earlier, otherwise Java garbage collection will release the storage eventually.
 *
 * <p>To extract concrete image, first check {@link StorageType} and then use the provided
 * extractors:
 *
 * <ul>
 *   <li>{@link ByteBufferExtractor}
 *   <li>{@link BitmapExtractor}
 *   <li>{@link MediaImageExtractor}
 * </ul>
 */
public class MPImage implements Closeable {

  /** Specifies the image format of an image. */
  @IntDef({
    IMAGE_FORMAT_UNKNOWN,
    IMAGE_FORMAT_RGBA,
    IMAGE_FORMAT_RGB,
    IMAGE_FORMAT_NV12,
    IMAGE_FORMAT_NV21,
    IMAGE_FORMAT_YV12,
    IMAGE_FORMAT_YV21,
    IMAGE_FORMAT_YUV_420_888,
    IMAGE_FORMAT_ALPHA,
    IMAGE_FORMAT_JPEG,
    IMAGE_FORMAT_VEC32F1,
    IMAGE_FORMAT_VEC32F2,
  })
  @Retention(RetentionPolicy.SOURCE)
  public @interface MPImageFormat {}

  public static final int IMAGE_FORMAT_UNKNOWN = 0;
  public static final int IMAGE_FORMAT_RGBA = 1;
  public static final int IMAGE_FORMAT_RGB = 2;
  public static final int IMAGE_FORMAT_NV12 = 3;
  public static final int IMAGE_FORMAT_NV21 = 4;
  public static final int IMAGE_FORMAT_YV12 = 5;
  public static final int IMAGE_FORMAT_YV21 = 6;
  public static final int IMAGE_FORMAT_YUV_420_888 = 7;
  public static final int IMAGE_FORMAT_ALPHA = 8;
  public static final int IMAGE_FORMAT_JPEG = 9;
  public static final int IMAGE_FORMAT_VEC32F1 = 10;
  public static final int IMAGE_FORMAT_VEC32F2 = 11;

  /** Specifies the image container type. Would be useful for choosing extractors. */
  @IntDef({
    STORAGE_TYPE_BITMAP,
    STORAGE_TYPE_BYTEBUFFER,
    STORAGE_TYPE_MEDIA_IMAGE,
    STORAGE_TYPE_IMAGE_PROXY,
  })
  @Retention(RetentionPolicy.SOURCE)
  public @interface StorageType {}

  public static final int STORAGE_TYPE_BITMAP = 1;
  public static final int STORAGE_TYPE_BYTEBUFFER = 2;
  public static final int STORAGE_TYPE_MEDIA_IMAGE = 3;
  public static final int STORAGE_TYPE_IMAGE_PROXY = 4;

  /**
   * Returns a list of supported image properties for this {@link MPImage}.
   *
   * <p>Currently {@link MPImage} only support single storage type so the size of return list will
   * always be 1.
   *
   * @see MPImageProperties
   */
  public List<MPImageProperties> getContainedImageProperties() {
    return Collections.singletonList(getContainer().getImageProperties());
  }

  /** Returns the timestamp attached to the image. */
  long getTimestamp() {
     return timestamp;
  }

  /** Returns the width of the image. */
  public int getWidth() {
    return width;
  }

  /** Returns the height of the image. */
  public int getHeight() {
    return height;
  }

  /** Acquires a reference on this {@link MPImage}. This will increase the reference count by 1. */
  private synchronized void acquire() {
    referenceCount += 1;
  }

  /**
   * Removes a reference that was previously acquired or init.
   *
   * <p>When {@link MPImage} is created, it has 1 reference count.
   *
   * <p>When the reference count becomes 0, it will release the resource under the hood.
   */
  @Override
  // TODO: Create an internal flag to indicate image is closed, or use referenceCount
  public synchronized void close() {
    referenceCount -= 1;
    if (referenceCount == 0) {
      for (MPImageContainer imageContainer : containerMap.values()) {
        imageContainer.close();
      }
    }
  }

  /** Advanced API access for {@link MPImage}. */
  static final class Internal {

    /**
     * Acquires a reference on this {@link MPImage}. This will increase the reference count by 1.
     *
     * <p>This method is more useful for image consumer to acquire a reference so image resource
     * will not be closed accidentally. As image creator, normal developer doesn't need to call this
     * method.
     *
     * <p>The reference count is 1 when {@link MPImage} is created. Developer can call {@link
     * #close()} to indicate it doesn't need this {@link MPImage} anymore.
     *
     * @see #close()
     */
    void acquire() {
      image.acquire();
    }

    private final MPImage image;

    // Only MPImage creates the internal helper.
    private Internal(MPImage image) {
      this.image = image;
    }
  }

  /** Gets {@link Internal} object which contains internal APIs. */
  Internal getInternal() {
    return new Internal(this);
  }

  private final Map<MPImageProperties, MPImageContainer> containerMap;
  private final long timestamp;
  private final int width;
  private final int height;

  private int referenceCount;

  /** Constructs a {@link MPImage} with a built container. */
  MPImage(MPImageContainer container, long timestamp, int width, int height) {
    this.containerMap = new HashMap<>();
    containerMap.put(container.getImageProperties(), container);
    this.timestamp = timestamp;
    this.width = width;
    this.height = height;
    this.referenceCount = 1;
  }

  /**
   * Gets one available container.
   *
   * @return the current container.
   */
  MPImageContainer getContainer() {
    // According to the design, in the future we will support multiple containers in one image.
    // Currently just return the original container.
    // TODO: Cache multiple containers in MPImage.
    return containerMap.values().iterator().next();
  }

  /**
   * Gets container from required {@code storageType}. Returns {@code null} if not existed.
   *
   * <p>If there are multiple containers with required {@code storageType}, returns the first one.
   */
  @Nullable
  MPImageContainer getContainer(@StorageType int storageType) {
    for (Entry<MPImageProperties, MPImageContainer> entry : containerMap.entrySet()) {
      if (entry.getKey().getStorageType() == storageType) {
        return entry.getValue();
      }
    }
    return null;
  }

  /** Gets container from required {@code imageProperties}. Returns {@code null} if non existed. */
  @Nullable
  MPImageContainer getContainer(MPImageProperties imageProperties) {
    return containerMap.get(imageProperties);
  }

  /** Adds a new container if it doesn't exist. Returns {@code true} if it succeeds. */
  boolean addContainer(MPImageContainer container) {
    MPImageProperties imageProperties = container.getImageProperties();
    if (containerMap.containsKey(imageProperties)) {
      return false;
    }
    containerMap.put(imageProperties, container);
    return true;
  }
}
