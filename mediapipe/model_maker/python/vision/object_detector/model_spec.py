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
"""Object detector model specification."""
import dataclasses
import enum
import functools
from typing import List

from mediapipe.model_maker.python.core.utils import file_util


MOBILENET_V2_FILES = file_util.DownloadedFiles(
    'object_detector/mobilenetv2',
    'https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i256_ckpt.tar.gz',
    is_folder=True,
)


@dataclasses.dataclass
class ModelSpec(object):
  """Specification of object detector model."""

  # Mean and Stddev image preprocessing normalization values.
  mean_norm = (0.5,)
  stddev_norm = (0.5,)
  mean_rgb = (127.5,)
  stddev_rgb = (127.5,)

  downloaded_files: file_util.DownloadedFiles
  input_image_shape: List[int]


mobilenet_v2_spec = functools.partial(
    ModelSpec,
    downloaded_files=MOBILENET_V2_FILES,
    input_image_shape=[256, 256, 3],
)


@enum.unique
class SupportedModels(enum.Enum):
  """Predefined object detector model specs supported by Model Maker."""

  MOBILENET_V2 = mobilenet_v2_spec

  @classmethod
  def get(cls, spec: 'SupportedModels') -> 'ModelSpec':
    """Get model spec from the input enum and initializes it."""
    if spec not in cls:
      raise TypeError(f'Unsupported object detector spec: {spec}')
    return spec.value()
