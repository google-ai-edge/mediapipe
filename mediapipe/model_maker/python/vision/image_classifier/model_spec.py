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
"""Image classifier model specification."""

import enum
import functools
from typing import List, Optional, Sequence


class ModelSpec(object):
  """Specification of image classifier model."""

  def __init__(
      self,
      uri: str,
      input_image_shape: Optional[List[int]] = None,
      name: str = '',
      mean_rgb: Optional[Sequence[float]] = None,
      stddev_rgb: Optional[Sequence[float]] = None,
  ):
    """Initializes a new instance of the image classifier `ModelSpec` class.

    Args:
      uri: str, URI to the pretrained model.
      input_image_shape: list of int, input image shape. Default: [224, 224].
      name: str, model spec name.
      mean_rgb: Normalizing mean RGB.
      stddev_rgb: Normalizing std RGB.
    """
    self.uri = uri
    self.name = name

    if input_image_shape is None:
      input_image_shape = [224, 224]
    self.input_image_shape = input_image_shape

    if mean_rgb is None:
      mean_rgb = [0.0]
    self.mean_rgb = mean_rgb

    if stddev_rgb is None:
      stddev_rgb = [255.0]
    self.stddev_rgb = stddev_rgb


mobilenet_v2_keras_spec = functools.partial(
    ModelSpec,
    uri=None,
    name='mobilenet_v2_keras',
    mean_rgb=[127.5],
    stddev_rgb=[128.0],
)

mobilenet_v2_spec = functools.partial(
    ModelSpec,
    uri='https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4',
    name='mobilenet_v2',
)

efficientnet_lite0_spec = functools.partial(
    ModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
    name='efficientnet_lite0',
)

efficientnet_lite2_spec = functools.partial(
    ModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
    input_image_shape=[260, 260],
    name='efficientnet_lite2',
)

efficientnet_lite4_spec = functools.partial(
    ModelSpec,
    uri='https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
    input_image_shape=[300, 300],
    name='efficientnet_lite4',
)


# TODO: Document the exposed models.
@enum.unique
class SupportedModels(enum.Enum):
  """Image classifier model supported by model maker."""

  MOBILENET_V2 = mobilenet_v2_spec
  MOBILENET_V2_KERAS = mobilenet_v2_keras_spec
  EFFICIENTNET_LITE0 = efficientnet_lite0_spec
  EFFICIENTNET_LITE2 = efficientnet_lite2_spec
  EFFICIENTNET_LITE4 = efficientnet_lite4_spec

  @classmethod
  def get(cls, spec: 'SupportedModels') -> 'ModelSpec':
    """Gets model spec from the input enum and initializes it."""
    if spec not in cls:
      raise TypeError('Unsupported image classifier spec: {}'.format(spec))

    return spec.value()
