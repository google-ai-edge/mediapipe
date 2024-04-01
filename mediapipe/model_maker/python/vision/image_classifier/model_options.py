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
"""Configurable model options for image classifier models."""

import dataclasses


@dataclasses.dataclass
class ImageClassifierModelOptions:
  """Configurable options for image classifier model.

  Attributes:
    dropout_rate: The fraction of the input units to drop, used in dropout
      layer.
    alpha: Float, larger than zero, controls the width of the network, Only used
      in `mobilenet_v2_keras_spec` model spec.
  """
  dropout_rate: float = 0.2
  alpha: float = 0.75
