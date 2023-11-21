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
import numpy as np
from typing import NamedTuple

from mediapipe.calculators.ovms import openvinoinferencecalculator_pb2
from mediapipe.calculators.ovms import openvinomodelserversessioncalculator_pb2
from mediapipe.python.solution_base import SolutionBase

_FULL_GRAPH_FILE_PATH = 'mediapipe/modules/ovms_modules/holistic_tracking_ovms.binarypb'

class OvmsHolisticTracking(SolutionBase):
  """Ovms Holistic Tracking.

  Ovms Holistic Tracking processes an input image frame returns output image frame
  with detected objects.
  """
  """
  Oryginal params in desktop example
  --calculator_graph_config_file /mediapipe/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt
  --input_video_path=/mediapipe/video.mp4
  --output_video_path=/mediapipe/output_holistic_ovms.mp4
  """
  def __init__(self):
    """Initializes a Ovms Holistic Tracking object.
    """
    super().__init__(
        binary_graph_path=_FULL_GRAPH_FILE_PATH)

  # input_video is the input_stream name from the graph
  def process(self, image: np.ndarray) -> NamedTuple:
    return super().process(input_data={'input_video': image})
