# Copyright 2024 The MediaPipe Authors.
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

"""Multi-person MediaPipe Holistic.

`solutions.holistic.Holistic` is single-person: its pose stage keeps only the
most prominent detection. `MultiHolistic` runs the `HolisticLandmarkMultiCpu`
graph, which detects every person in the frame and runs pose + hands + face for
each one, returning a list (one entry per person).

Example:

    import cv2, mediapipe as mp
    mph = mp.solutions.holistic_multi.MultiHolistic()
    image = cv2.cvtColor(cv2.imread('group.jpg'), cv2.COLOR_BGR2RGB)
    result = mph.process(image)
    for pose in result.pose_landmarks:        # one NormalizedLandmarkList / person
        mp.solutions.drawing_utils.draw_landmarks(
            image, pose, mp.solutions.holistic.POSE_CONNECTIONS)
    mph.close()
"""

import collections
import os
from typing import List, NamedTuple

import numpy as np

from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import calculator_graph
from mediapipe.python._framework_bindings import image_frame as image_frame_lib
from mediapipe.python._framework_bindings import resource_util

_BINARY_GRAPH_PATH = (
    'mediapipe/modules/holistic_landmark/holistic_landmark_multi_cpu.binarypb'
)
_OUTPUT_STREAMS = (
    'multi_pose_landmarks', 'multi_left_hand_landmarks',
    'multi_right_hand_landmarks', 'multi_face_landmarks',
)
_MultiHolisticResults = collections.namedtuple(
    'MultiHolisticResults',
    ['pose_landmarks', 'left_hand_landmarks', 'right_hand_landmarks',
     'face_landmarks'])


class MultiHolistic:
  """Runs MediaPipe Holistic for every person detected in an image."""

  def __init__(self, model_complexity: int = 1,
               refine_face_landmarks: bool = False):
    """Initializes a MultiHolistic object.

    Args:
      model_complexity: Complexity of the pose landmark model: 0, 1 or 2.
      refine_face_landmarks: Whether to refine face landmarks around the eyes
        and lips and output additional iris landmarks.
    """
    root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-4])
    resource_util.set_resource_dir(root)
    self._graph = calculator_graph.CalculatorGraph(
        binary_graph_path=os.path.join(root, _BINARY_GRAPH_PATH))
    self._latest = {}
    for stream in _OUTPUT_STREAMS:
      self._graph.observe_output_stream(stream, self._make_callback(stream),
                                        True)
    self._graph.start_run({
        'model_complexity': packet_creator.create_int(model_complexity),
        'refine_face_landmarks':
            packet_creator.create_bool(refine_face_landmarks),
    })
    self._timestamp = 0

  def _make_callback(self, stream):
    def callback(_, packet):
      self._latest[stream] = packet
    return callback

  def _collect(self, stream) -> List[landmark_pb2.NormalizedLandmarkList]:
    packet = self._latest.get(stream)
    if packet is None or packet.is_empty():
      return []
    return list(packet_getter.get_proto_list(packet))

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image; returns per-person landmark lists.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Returns:
      A namedtuple with `pose_landmarks`, `left_hand_landmarks`,
      `right_hand_landmarks` and `face_landmarks`, each a list of
      `NormalizedLandmarkList` (one entry per detected person; hand/face entries
      may be empty when not visible for that person).
    """
    self._latest.clear()
    self._timestamp += 1
    self._graph.add_packet_to_input_stream(
        stream='image',
        packet=packet_creator.create_image_frame(
            image_frame_lib.ImageFrame(
                image_format=image_frame_lib.ImageFormat.SRGB,
                data=np.ascontiguousarray(image))),
        timestamp=self._timestamp)
    self._graph.wait_until_idle()
    return _MultiHolisticResults(*[self._collect(s) for s in _OUTPUT_STREAMS])

  def close(self) -> None:
    self._graph.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()
