# Copyright 2021 The MediaPipe Authors.
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
"""MediaPipe Selfie Segmentation."""

from typing import NamedTuple

import numpy as np
# The following imports are needed because python pb2 silently discards
# unknown protobuf fields.
# pylint: disable=unused-import
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_segmentation_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2
# pylint: enable=unused-import

from mediapipe.python.solution_base import SolutionBase

BINARYPB_FILE_PATH = 'mediapipe/modules/selfie_segmentation/selfie_segmentation_cpu.binarypb'


class SelfieSegmentation(SolutionBase):
  """MediaPipe Selfie Segmentation.

  MediaPipe Selfie Segmentation processes an RGB image and returns a
  segmentation mask.

  Please refer to
  https://solutions.mediapipe.dev/selfie_segmentation#python-solution-api for
  usage examples.
  """

  def __init__(self, model_selection=0):
    """Initializes a MediaPipe Selfie Segmentation object.

    Args:
      model_selection: 0 or 1. 0 to select a general-purpose model, and 1 to
        select a model more optimized for landscape images. See details in
        https://solutions.mediapipe.dev/selfie_segmentation#model_selection.
    """
    super().__init__(
        binary_graph_path=BINARYPB_FILE_PATH,
        side_inputs={
            'model_selection': model_selection,
        },
        outputs=['segmentation_mask'])

  def process(self, image: np.ndarray) -> NamedTuple:
    """Processes an RGB image and returns a segmentation mask.

    Args:
      image: An RGB image represented as a numpy ndarray.

    Raises:
      RuntimeError: If the underlying graph throws any error.
      ValueError: If the input image is not three channel RGB.

    Returns:
      A NamedTuple object with a "segmentation_mask" field that contains a float
      type 2d np array representing the mask.
    """

    return super().process(input_data={'image': image})
