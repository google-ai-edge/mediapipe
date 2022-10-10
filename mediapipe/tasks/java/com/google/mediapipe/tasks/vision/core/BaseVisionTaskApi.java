// Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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

package com.google.mediapipe.tasks.vision.core;

import android.graphics.RectF;
import com.google.mediapipe.formats.proto.RectProto.NormalizedRect;
import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.ProtoUtil;
import com.google.mediapipe.framework.image.Image;
import com.google.mediapipe.tasks.core.TaskResult;
import com.google.mediapipe.tasks.core.TaskRunner;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/** The base class of MediaPipe vision tasks. */
public class BaseVisionTaskApi implements AutoCloseable {
  private static final long MICROSECONDS_PER_MILLISECOND = 1000;
  private final TaskRunner runner;
  private final RunningMode runningMode;
  private final String imageStreamName;
  private final Optional<String> normRectStreamName;

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
    ProtoUtil.registerTypeName(NormalizedRect.class, "mediapipe.NormalizedRect");
  }

  /**
   * Constructor to initialize a {@link BaseVisionTaskApi} only taking images as input.
   *
   * @param runner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   * @param imageStreamName the name of the input image stream.
   */
  public BaseVisionTaskApi(TaskRunner runner, RunningMode runningMode, String imageStreamName) {
    this.runner = runner;
    this.runningMode = runningMode;
    this.imageStreamName = imageStreamName;
    this.normRectStreamName = Optional.empty();
  }

  /**
   * Constructor to initialize a {@link BaseVisionTaskApi} taking images and normalized rects as
   * input.
   *
   * @param runner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   * @param imageStreamName the name of the input image stream.
   * @param normRectStreamName the name of the input normalized rect image stream.
   */
  public BaseVisionTaskApi(
      TaskRunner runner,
      RunningMode runningMode,
      String imageStreamName,
      String normRectStreamName) {
    this.runner = runner;
    this.runningMode = runningMode;
    this.imageStreamName = imageStreamName;
    this.normRectStreamName = Optional.of(normRectStreamName);
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @throws MediaPipeException if the task is not in the image mode or requires a normalized rect
   *     input.
   */
  protected TaskResult processImageData(Image image) {
    if (runningMode != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the image mode. Current running mode:"
              + runningMode.name());
    }
    if (normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task expects a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    return runner.process(inputPackets);
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @param roi a {@link RectF} defining the region-of-interest to process in the image. Coordinates
   *     are expected to be specified as normalized values in [0,1].
   * @throws MediaPipeException if the task is not in the image mode or doesn't require a normalized
   *     rect.
   */
  protected TaskResult processImageData(Image image, RectF roi) {
    if (runningMode != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the image mode. Current running mode:"
              + runningMode.name());
    }
    if (!normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task doesn't expect a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    inputPackets.put(
        normRectStreamName.get(),
        runner.getPacketCreator().createProto(convertToNormalizedRect(roi)));
    return runner.process(inputPackets);
  }

  /**
   * A synchronous method to process continuous video frames. The call blocks the current thread
   * until a failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode or requires a normalized rect
   *     input.
   */
  protected TaskResult processVideoData(Image image, long timestampMs) {
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the video mode. Current running mode:"
              + runningMode.name());
    }
    if (normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task expects a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    return runner.process(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /**
   * A synchronous method to process continuous video frames. The call blocks the current thread
   * until a failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @param roi a {@link RectF} defining the region-of-interest to process in the image. Coordinates
   *     are expected to be specified as normalized values in [0,1].
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode or doesn't require a normalized
   *     rect.
   */
  protected TaskResult processVideoData(Image image, RectF roi, long timestampMs) {
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the video mode. Current running mode:"
              + runningMode.name());
    }
    if (!normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task doesn't expect a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    inputPackets.put(
        normRectStreamName.get(),
        runner.getPacketCreator().createProto(convertToNormalizedRect(roi)));
    return runner.process(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /**
   * An asynchronous method to send live stream data to the {@link TaskRunner}. The results will be
   * available in the user-defined result listener.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode or requires a normalized rect
   *     input.
   */
  protected void sendLiveStreamData(Image image, long timestampMs) {
    if (runningMode != RunningMode.LIVE_STREAM) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the live stream mode. Current running mode:"
              + runningMode.name());
    }
    if (normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task expects a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    runner.send(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /**
   * An asynchronous method to send live stream data to the {@link TaskRunner}. The results will be
   * available in the user-defined result listener.
   *
   * @param image a MediaPipe {@link Image} object for processing.
   * @param roi a {@link RectF} defining the region-of-interest to process in the image. Coordinates
   *     are expected to be specified as normalized values in [0,1].
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode or doesn't require a normalized
   *     rect.
   */
  protected void sendLiveStreamData(Image image, RectF roi, long timestampMs) {
    if (runningMode != RunningMode.LIVE_STREAM) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the live stream mode. Current running mode:"
              + runningMode.name());
    }
    if (!normRectStreamName.isPresent()) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task doesn't expect a normalized rect as input.");
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    inputPackets.put(
        normRectStreamName.get(),
        runner.getPacketCreator().createProto(convertToNormalizedRect(roi)));
    runner.send(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /** Closes and cleans up the MediaPipe vision task. */
  @Override
  public void close() {
    runner.close();
  }

  /** Converts a {@link RectF} object into a {@link NormalizedRect} protobuf message. */
  private static NormalizedRect convertToNormalizedRect(RectF rect) {
    return NormalizedRect.newBuilder()
        .setXCenter(rect.centerX())
        .setYCenter(rect.centerY())
        .setWidth(rect.width())
        .setHeight(rect.height())
        .build();
  }
}
