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
"""Configurable model options for gesture recognizer models."""

import dataclasses
from typing import List


@dataclasses.dataclass
class GestureRecognizerModelOptions:
  """Configurable options for gesture recognizer model.

  Attributes:
    dropout_rate: The fraction of the input units to drop, used in dropout
      layer.
    layer_widths: A list of hidden layer widths for the gesture model. Each
      element in the list will create a new hidden layer with the specified
      width. The hidden layers are separated with BatchNorm, Dropout, and ReLU.
      Defaults to an empty list(no hidden layers).
  """
  dropout_rate: float = 0.05
  layer_widths: List[int] = dataclasses.field(default_factory=list)
