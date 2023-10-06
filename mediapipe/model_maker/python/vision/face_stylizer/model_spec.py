# Copyright 2023 The MediaPipe Authors.
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
"""Face stylizer model specification."""

import enum
import functools
from typing import List


class ModelSpec(object):
  """Specification of face stylizer model."""

  mean_rgb = [127.5]
  stddev_rgb = [127.5]

  def __init__(
      self, style_block_num: int, input_image_shape: List[int], name: str = ''
  ):
    """Initializes a new instance of the `ModelSpec` class for face stylizer.

    Args:
      style_block_num: int, number of style block in the decoder.
      input_image_shape: list of int, input image shape.
      name: str, model spec name.
    """
    self.style_block_num = style_block_num
    self.input_image_shape = input_image_shape
    self.name = name


blaze_face_stylizer_256_spec = functools.partial(
    ModelSpec,
    style_block_num=12,
    input_image_shape=[256, 256],
    name='blaze_face_stylizer_256',
)


# TODO: Document the exposed models.
@enum.unique
class SupportedModels(enum.Enum):
  """Face stylizer model supported by MediaPipe model maker."""

  BLAZE_FACE_STYLIZER_256 = blaze_face_stylizer_256_spec

  @classmethod
  def get(cls, spec: 'SupportedModels') -> 'ModelSpec':
    """Gets model spec from the input enum and initializes it."""
    if spec not in cls:
      raise TypeError('Unsupported face stylizer spec: {}'.format(spec))

    return spec.value()
