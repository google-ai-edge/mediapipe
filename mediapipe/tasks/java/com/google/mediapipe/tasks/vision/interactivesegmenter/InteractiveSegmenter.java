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

package com.google.mediapipe.tasks.vision.interactivesegmenter;

import android.content.Context;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.AndroidPacketCreator;
import com.google.mediapipe.framework.Graph;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint;
import com.google.mediapipe.tasks.core.BaseOptionsUtils;
import com.google.mediapipe.tasks.core.proto.BaseOptionsProto;
import com.google.mediapipe.tasks.vision.interactivesegmenter.proto.StrokeProto;
import com.google.mediapipe.tasks.vision.interactivesegmenter.proto.StrokeProto.Stroke.BrushMode;
import com.google.mediapipe.tasks.vision.interactivesegmenter.proto.StrokeProto.Stroke.Point;
import com.google.mediapipe.tasks.vision.interactivesegmenter.proto.StrokeProto.Strokes;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Objects;

/**
 * Performs interactive segmentation on images in two steps:
 *
 * <ol>
 *   <li>Set an image with `setImage`.
 *   <li>Segment with `segment` one or more times efficiently providing various strokes, allowing
 *       the user to fine tune the produced segmentation mask.
 * </ol>
 */
public final class InteractiveSegmenter implements AutoCloseable {
  private final long nativeSegmenterHandle;
  private final Graph graph;
  private final AndroidPacketCreator packetCreator;

  static {
    System.loadLibrary("mediapipe_tasks_jni");
  }

  private InteractiveSegmenter(long nativeSegmenterHandle) {
    this.nativeSegmenterHandle = nativeSegmenterHandle;
    this.graph = new Graph();
    this.packetCreator = new AndroidPacketCreator(this.graph);
  }

  /**
   * Creates an {@link InteractiveSegmenter} instance.
   *
   * @param context an Android {@link Context}.
   * @param options an {@link InteractiveSegmenterOptions} instance.
   * @throws MediaPipeException if there is an error during creation.
   */
  public static InteractiveSegmenter createFromOptions(
      Context context, InteractiveSegmenterOptions options) {
    AndroidAssetUtil.initializeNativeAssetManager(context);
    BaseOptionsProto.BaseOptions baseOptionsProto =
        BaseOptionsUtils.convertBaseOptionsToProto(options.baseOptions());
    byte[] baseOptionsBytes = baseOptionsProto.toByteArray();
    String appVersion = BaseOptionsUtils.getAppVersion(context);
    String appId = BaseOptionsUtils.getAppId(context);
    long nativeSegmenterHandle =
        nativeCreate(
            baseOptionsBytes,
            appId,
            appVersion,
            BaseOptionsUtils.HOST_ENVIRONMENT_ANDROID,
            BaseOptionsUtils.HOST_SYSTEM_ANDROID);
    if (nativeSegmenterHandle == 0) {
      // TODO: b/509975940 - get rid of ordinal() use.
      @SuppressWarnings("EnumOrdinal")
      final int code = MediaPipeException.StatusCode.INTERNAL.ordinal();
      throw new MediaPipeException(code, "Error creating InteractiveSegmenter. Handle is 0.");
    }
    return new InteractiveSegmenter(nativeSegmenterHandle);
  }

  /** Sets the image to be segmented. */
  public void setImage(MPImage image) {
    Objects.requireNonNull(image, "Image must not be null");
    Packet imagePacket = packetCreator.createImage(image);
    try {
      nativeSetImage(nativeSegmenterHandle, imagePacket.getNativeHandle());
    } finally {
      imagePacket.release();
    }
  }

  /**
   * Performs segmentation using the provided strokes.
   *
   * @param strokes The List of {@link Stroke}s containing the user selected points/strokes.
   * @return An {@link MPImage} representing the segmentation mask.
   * @throws MediaPipeException if there is an error during segmentation.
   */
  public MPImage segment(List<Stroke> strokes) {
    Objects.requireNonNull(strokes, "Strokes must not be null");
    byte[] strokesBytes = convertToProto(strokes).toByteArray();
    long resultPacketHandle =
        nativeSegment(nativeSegmenterHandle, strokesBytes, graph.getNativeHandle());
    if (resultPacketHandle == 0) {
      // TODO: b/509975940 - get rid of ordinal() use.
      @SuppressWarnings("EnumOrdinal")
      final int code = MediaPipeException.StatusCode.INTERNAL.ordinal();
      throw new MediaPipeException(code, "Error running segmentation.");
    }
    Packet resultPacket = Packet.create(resultPacketHandle);
    try {
      int width = PacketGetter.getImageWidth(resultPacket);
      int height = PacketGetter.getImageHeight(resultPacket);
      ByteBuffer buffer = PacketGetter.getImageDataDirectly(resultPacket);
      if (buffer == null) {
        // TODO: b/509975940 - get rid of ordinal() use.
        @SuppressWarnings("EnumOrdinal")
        final int code = MediaPipeException.StatusCode.INTERNAL.ordinal();
        throw new MediaPipeException(code, "Error getting output image data.");
      }

      // Copy the buffer because the underlying native memory of `resultPacket`
      // will be released and invalidated in the finally block.
      ByteBuffer bufferCopy = ByteBuffer.allocateDirect(buffer.capacity());
      bufferCopy.put(buffer);
      bufferCopy.rewind();

      ByteBufferImageBuilder builder =
          new ByteBufferImageBuilder(bufferCopy, width, height, MPImage.IMAGE_FORMAT_VEC32F1);
      return builder.build();
    } finally {
      resultPacket.release();
    }
  }

  private static Strokes convertToProto(List<Stroke> strokes) {
    Strokes.Builder strokesProtoBuilder = Strokes.newBuilder();
    for (Stroke stroke : strokes) {
      StrokeProto.Stroke.Builder strokeProtoBuilder =
          StrokeProto.Stroke.newBuilder()
              .setBrushMode(BrushMode.forNumber(stroke.brushMode().getValue()));
      for (NormalizedKeypoint point : stroke.points()) {
        strokeProtoBuilder.addPoint(Point.newBuilder().setX(point.x()).setY(point.y()).build());
      }
      strokeProtoBuilder.setIsCompleted(stroke.completed());
      strokesProtoBuilder.addStroke(strokeProtoBuilder.build());
    }
    return strokesProtoBuilder.build();
  }

  @Override
  public void close() {
    nativeClose(nativeSegmenterHandle);
    graph.tearDown();
  }

  private static native long nativeCreate(
      byte[] baseOptionsBytes,
      String appId,
      String appVersion,
      int hostEnvironment,
      int hostSystem);

  private native void nativeSetImage(long segmenterHandle, long imagePacketHandle);

  private native long nativeSegment(long segmenterHandle, byte[] strokesBytes, long graphHandle);

  private native void nativeClose(long segmenterHandle);
}
