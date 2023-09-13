# Copyright 2022 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Data utility library."""

import cv2
import numpy as np
import tensorflow as tf


def load_image(path: str) -> np.ndarray:
  """Loads an image as an RGB numpy array.

  Args:
    path: input image file absolute path.

  Returns:
    An RGB image in numpy.ndarray.
  """
  tf.compat.v1.logging.info('Loading RGB image %s', path)
  # TODO Replace the OpenCV image load and conversion library by
  # MediaPipe image utility library once it is ready.
  image = cv2.imread(path)
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
