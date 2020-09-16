# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""MediaPipe upper body pose tracker.

MediaPipe upper body pose tracker takes an RGB image as the input and returns
a pose landmark list and an annotated RGB image represented as a numpy ndarray.

Usage examples:
  pose_tracker = UpperBodyPoseTracker()

  pose_landmarks, _ = pose_tracker.run(
      input_file='/tmp/input.png',
      output_file='/tmp/output.png')

  input_image = cv2.imread('/tmp/input.png')[:, :, ::-1]
  pose_landmarks, annotated_image = pose_tracker.run(input_image)

  pose_tracker.run_live()

  pose_tracker.close()
"""

import os
import time
from typing import Tuple, Union

import cv2
import mediapipe.python as mp
import numpy as np
# resources dependency
from mediapipe.framework.formats import landmark_pb2

# Input and output stream names.
INPUT_VIDEO = 'input_video'
OUTPUT_VIDEO = 'output_video'
POSE_LANDMARKS = 'pose_landmarks'


class UpperBodyPoseTracker:
  """MediaPipe upper body pose tracker."""

  def __init__(self):
    """The init method of MediaPipe upper body pose tracker.

    The method reads the upper body pose tracking cpu binary graph and
    initializes a CalculatorGraph from it. The output packets of pose_landmarks
    and output_video output streams will be observed by callbacks. The graph
    will be started at the end of this method, waiting for input packets.
    """
    # MediaPipe package root path
    root_path = os.sep.join( os.path.abspath(__file__).split(os.sep)[:-4])
    mp.resource_util.set_resource_dir(root_path)

    self._graph = mp.CalculatorGraph(
        binary_graph_path=os.path.join(
            root_path,
            'mediapipe/graphs/pose_tracking/upper_body_pose_tracking_cpu.binarypb'
        ))
    self._outputs = {}
    for stream_name in [POSE_LANDMARKS, OUTPUT_VIDEO]:
      self._graph.observe_output_stream(stream_name, self._assign_packet)
    self._graph.start_run()

  def run(
      self,
      input_frame: np.ndarray = None,
      *,
      input_file: str = None,
      output_file: str = None
  ) -> Tuple[Union[None, landmark_pb2.NormalizedLandmarkList], np.ndarray]:
    """The run method of MediaPipe upper body pose tracker.

    MediaPipe upper body pose tracker takes either the path to an image file or
    an RGB image represented as a numpy ndarray and it returns the pose
    landmarks list and the annotated RGB image represented as a numpy ndarray.

    Args:
      input_frame: An RGB image represented as a numpy ndarray.
      input_file: The path to an image file.
      output_file: The file path that the annotated image will be saved into.

    Returns:
      pose_landmarks: The pose landmarks list.
      annotated_image: The image with pose landmarks annotations.

    Raises:
      RuntimeError: If the input frame doesn't contain 3 channels (RGB format)
        or the input arg is not correctly provided.

    Examples
      pose_tracker = UpperBodyPoseTracker()
      pose_landmarks, _ = pose_tracker.run(
          input_file='/tmp/input.png',
          output_file='/tmp/output.png')

      # Read an image and convert the BGR image to RGB.
      input_image = cv2.cvtColor(cv2.imread('/tmp/input.png'), COLOR_BGR2RGB)
      pose_landmarks, annotated_image = pose_tracker.run(input_image)
      pose_tracker.close()
    """
    if input_file is None and input_frame is None:
      raise RuntimeError(
          'Must provide either a path to an image file or an RGB image represented as a numpy.ndarray.'
      )

    if input_file:
      if input_frame is not None:
        raise RuntimeError(
            'Must only provide either \'input_file\' or \'input_frame\'.')
      else:
        input_frame = cv2.imread(input_file)[:, :, ::-1]

    pose_landmarks, annotated_image = self._run_graph(input_frame)
    if output_file:
      cv2.imwrite(output_file, annotated_image[:, :, ::-1])
    return pose_landmarks, annotated_image

  def run_live(self) -> None:
    """Run MediaPipe upper body pose tracker with live camera input.

    The method will be self-terminated after 30 seconds. If you need to
    terminate it earlier, press the Esc key to stop the run manually. Note that
    you need to select the output image window rather than the terminal window
    first and then press the key.

    Examples:
      pose_tracker = UpperBodyPoseTracker()
      pose_tracker.run_live()
      pose_tracker.close()
    """
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    print(
        'Press Esc within the output image window to stop the run, or let it '
        'self terminate after 30 seconds.')
    while cap.isOpened() and time.time() - start_time < 30:
      success, input_frame = cap.read()
      if not success:
        break
      input_frame = cv2.cvtColor(cv2.flip(input_frame, 1), cv2.COLOR_BGR2RGB)
      input_frame.flags.writeable = False
      _, output_frame = self._run_graph(input_frame)
      cv2.imshow('MediaPipe upper body pose tracker',
                 cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
      if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()
    cv2.destroyAllWindows()

  def close(self) -> None:
    self._graph.close()
    self._graph = None
    self._outputs = None

  def _run_graph(
      self,
      input_frame: np.ndarray = None,
  ) -> Tuple[Union[None, landmark_pb2.NormalizedLandmarkList], np.ndarray]:
    """The internal run graph method.

    Args:
      input_frame: An RGB image represented as a numpy ndarray.

    Returns:
      pose_landmarks: The pose landmarks list.
      annotated_image: The image with pose landmarks annotations.

    Raises:
      RuntimeError: If the input frame doesn't contain 3 channels representing
      RGB.
    """

    if input_frame.shape[2] != 3:
      raise RuntimeError('input frame must have 3 channels.')

    self._outputs.clear()
    start_time = time.time()
    self._graph.add_packet_to_input_stream(
        stream=INPUT_VIDEO,
        packet=mp.packet_creator.create_image_frame(
            image_format=mp.ImageFormat.SRGB, data=input_frame),
        timestamp=mp.Timestamp.from_seconds(start_time))
    self._graph.wait_until_idle()

    pose_landmarks = None
    if POSE_LANDMARKS in self._outputs:
      pose_landmarks = mp.packet_getter.get_proto(self._outputs[POSE_LANDMARKS])
    annotated_image = mp.packet_getter.get_image_frame(
        self._outputs[OUTPUT_VIDEO]).numpy_view()
    print('UpperBodyPoseTracker.Run() took',
          time.time() - start_time, 'seconds')
    return pose_landmarks, annotated_image

  def _assign_packet(self, stream_name: str, packet: mp.Packet) -> None:
    self._outputs[stream_name] = packet
