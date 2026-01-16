// Copyright 2022 The MediaPipe Authors.
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
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.TaskResult;
import com.google.mediapipe.tasks.core.TaskRunner;
import java.util.HashMap;
import java.util.Map;

/** The base class of MediaPipe vision tasks. */
public class BaseVisionTaskApi implements AutoCloseable {
  protected static final long MICROSECONDS_PER_MILLISECOND = 1000;
  protected final TaskRunner runner;
  protected final RunningMode runningMode;
  protected final String imageStreamName;
  protected final String normRectStreamName;

  static {
    ProtoUtil.registerTypeName(NormalizedRect.class, "mediapipe.NormalizedRect");
  }

  /**
   * Constructor to initialize a {@link BaseVisionTaskApi}.
   *
   * @param runner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   * @param imageStreamName the name of the input image stream.
   * @param normRectStreamName the name of the input normalized rect image stream used to provide
   *     (mandatory) rotation and (optional) region-of-interest.
   */
  public BaseVisionTaskApi(
      TaskRunner runner,
      RunningMode runningMode,
      String imageStreamName,
      String normRectStreamName) {
    this.runner = runner;
    this.runningMode = runningMode;
    this.imageStreamName = imageStreamName;
    this.normRectStreamName = normRectStreamName;
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @throws MediaPipeException if the task is not in the image mode.
   */
  protected TaskResult processImageData(
      MPImage image, ImageProcessingOptions imageProcessingOptions) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    if (!normRectStreamName.isEmpty()) {
      inputPackets.put(
          normRectStreamName,
          runner
              .getPacketCreator()
              .createProto(convertToNormalizedRect(imageProcessingOptions, image)));
    }
    return processImageData(inputPackets);
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * @param inputPackets the maps of input stream names to the input packets.
   * @throws MediaPipeException if the task is not in the image mode.
   */
  protected TaskResult processImageData(Map<String, Packet> inputPackets) {
    if (runningMode != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the image mode. Current running mode:"
              + runningMode.name());
    }
    return runner.process(inputPackets);
  }

  /**
   * A synchronous method to process continuous video frames. The call blocks the current thread
   * until a failure status or a successful result is returned.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode.
   */
  protected TaskResult processVideoData(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    if (!normRectStreamName.isEmpty()) {
      inputPackets.put(
          normRectStreamName,
          runner
              .getPacketCreator()
              .createProto(convertToNormalizedRect(imageProcessingOptions, image)));
    }
    return processVideoData(inputPackets, timestampMs);
  }

  /**
   * A synchronous method to process continuous video frames. The call blocks the current thread
   * until a failure status or a successful result is returned.
   *
   * @param inputPackets the maps of input stream names to the input packets.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode.
   */
  protected TaskResult processVideoData(Map<String, Packet> inputPackets, long timestampMs) {
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the video mode. Current running mode:"
              + runningMode.name());
    }
    return runner.process(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /**
   * An asynchronous method to send live stream data to the {@link TaskRunner}. The results will be
   * available in the user-defined result listener.
   *
   * @param image a MediaPipe {@link MPImage} object for processing.
   * @param imageProcessingOptions the {@link ImageProcessingOptions} specifying how to process the
   *     input image before running inference.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the stream mode.
   */
  protected void sendLiveStreamData(
      MPImage image, ImageProcessingOptions imageProcessingOptions, long timestampMs) {
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    if (!normRectStreamName.isEmpty()) {
      inputPackets.put(
          normRectStreamName,
          runner
              .getPacketCreator()
              .createProto(convertToNormalizedRect(imageProcessingOptions, image)));
    }
    sendLiveStreamData(inputPackets, timestampMs);
  }

  /**
   * An asynchronous method to send live stream data to the {@link TaskRunner}. The results will be
   * available in the user-defined result listener.
   *
   * @param inputPackets the maps of input stream names to the input packets.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the stream mode.
   */
  protected void sendLiveStreamData(Map<String, Packet> inputPackets, long timestampMs) {
    if (runningMode != RunningMode.LIVE_STREAM) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the live stream mode. Current running mode:"
              + runningMode.name());
    }
    runner.send(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /** Closes and cleans up the MediaPipe vision task. */
  @Override
  public void close() {
    runner.close();
  }

  /**
   * Converts an {@link ImageProcessingOptions} instance into a {@link NormalizedRect} protobuf
   * message.
   */
  protected static NormalizedRect convertToNormalizedRect(
      ImageProcessingOptions imageProcessingOptions, MPImage image) {
    RectF regionOfInterest =
        imageProcessingOptions.regionOfInterest().isPresent()
            ? imageProcessingOptions.regionOfInterest().get()
            : new RectF(0, 0, 1, 1);
    // For 90° and 270° rotations, we need to swap width and height.
    // This is due to the internal behavior of ImageToTensorCalculator, which:
    // - first denormalizes the provided rect by multiplying the rect width or
    //   height by the image width or height, respectively.
    // - then rotates this by denormalized rect by the provided rotation, and
    //   uses this for cropping,
    // - then finally rotates this back.
    boolean requiresSwap = imageProcessingOptions.rotationDegrees() % 180 != 0;
    return NormalizedRect.newBuilder()
        .setXCenter(regionOfInterest.centerX())
        .setYCenter(regionOfInterest.centerY())
        .setWidth(
            requiresSwap
                ? regionOfInterest.height() * image.getHeight() / image.getWidth()
                : regionOfInterest.width())
        .setHeight(
            requiresSwap
                ? regionOfInterest.width() * image.getWidth() / image.getHeight()
                : regionOfInterest.height())
        // Convert to radians anti-clockwise.
        .setRotation(-(float) Math.PI * imageProcessingOptions.rotationDegrees() / 180.0f)
        .build();
  }

  /**
   * Generates the timestamp of a vision task result object from the vision task running mode and
   * the output packet.
   *
   * @param runningMode MediaPipe Vision Tasks {@link RunningMode}.
   * @param packet the output {@link Packet}.
   */
  public static long generateResultTimestampMs(RunningMode runningMode, Packet packet) {
    if (runningMode == RunningMode.IMAGE) {
      return -1;
    }
    return packet.getTimestamp() / MICROSECONDS_PER_MILLISECOND;
  }
}
