# Copyright (c) 2023 Intel Corporation
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
#
"""Ovms Object Detection."""

from mediapipe.calculators.ovms import openvinoinferencecalculator_pb2
from mediapipe.calculators.ovms import openvinomodelserversessioncalculator_pb2
from mediapipe.python.solution_base import SolutionBase

_FULL_GRAPH_FILE_PATH = 'mediapipe/modules/ovms_modules/object_detection_ovms.binarypb'

class OvmsObjectDetection(SolutionBase):
  """Ovms Object Detection.

  Ovms Object Detection processes an input video returns output video
  with detectec objects.
  """
  """
  Oryginal params in desktop example
  --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_ovms1_graph.pbtxt
  --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/tested_video.mp4
  """
  def __init__(self,
              side_inputs=
              {'input_video_path':'/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4',
              'output_video_path':'/mediapipe/tested_video.mp4'}):
    """Initializes a Ovms Object Detection object.
    """
    super().__init__(
        binary_graph_path=_FULL_GRAPH_FILE_PATH,
        side_inputs=side_inputs)

  def process(self):
    self._graph.wait_until_done()
    return None
