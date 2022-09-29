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

import com.google.mediapipe.framework.MediaPipeException;
import com.google.mediapipe.framework.Packet;
import com.google.mediapipe.framework.image.Image;
import com.google.mediapipe.tasks.core.TaskResult;
import com.google.mediapipe.tasks.core.TaskRunner;
import java.util.HashMap;
import java.util.Map;

/** The base class of MediaPipe vision tasks. */
public class BaseVisionTaskApi implements AutoCloseable {
  private static final long MICROSECONDS_PER_MILLISECOND = 1000;
  private final TaskRunner runner;
  private final RunningMode runningMode;

  static {
    System.loadLibrary("mediapipe_tasks_vision_jni");
  }

  /**
   * Constructor to initialize an {@link BaseVisionTaskApi} from a {@link TaskRunner} and a vision
   * task {@link RunningMode}.
   *
   * @param runner a {@link TaskRunner}.
   * @param runningMode a mediapipe vision task {@link RunningMode}.
   */
  public BaseVisionTaskApi(TaskRunner runner, RunningMode runningMode) {
    this.runner = runner;
    this.runningMode = runningMode;
  }

  /**
   * A synchronous method to process single image inputs. The call blocks the current thread until a
   * failure status or a successful result is returned.
   *
   * @param imageStreamName the image input stream name.
   * @param image a MediaPipe {@link Image} object for processing.
   * @throws MediaPipeException if the task is not in the image mode.
   */
  protected TaskResult processImageData(String imageStreamName, Image image) {
    if (runningMode != RunningMode.IMAGE) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the image mode. Current running mode:"
              + runningMode.name());
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    return runner.process(inputPackets);
  }

  /**
   * A synchronous method to process continuous video frames. The call blocks the current thread
   * until a failure status or a successful result is returned.
   *
   * @param imageStreamName the image input stream name.
   * @param image a MediaPipe {@link Image} object for processing.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode.
   */
  protected TaskResult processVideoData(String imageStreamName, Image image, long timestampMs) {
    if (runningMode != RunningMode.VIDEO) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the video mode. Current running mode:"
              + runningMode.name());
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    return runner.process(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /**
   * An asynchronous method to send live stream data to the {@link TaskRunner}. The results will be
   * available in the user-defined result listener.
   *
   * @param imageStreamName the image input stream name.
   * @param image a MediaPipe {@link Image} object for processing.
   * @param timestampMs the corresponding timestamp of the input image in milliseconds.
   * @throws MediaPipeException if the task is not in the video mode.
   */
  protected void sendLiveStreamData(String imageStreamName, Image image, long timestampMs) {
    if (runningMode != RunningMode.LIVE_STREAM) {
      throw new MediaPipeException(
          MediaPipeException.StatusCode.FAILED_PRECONDITION.ordinal(),
          "Task is not initialized with the live stream mode. Current running mode:"
              + runningMode.name());
    }
    Map<String, Packet> inputPackets = new HashMap<>();
    inputPackets.put(imageStreamName, runner.getPacketCreator().createImage(image));
    runner.send(inputPackets, timestampMs * MICROSECONDS_PER_MILLISECOND);
  }

  /** Closes and cleans up the MediaPipe vision task. */
  @Override
  public void close() {
    runner.close();
  }
}
