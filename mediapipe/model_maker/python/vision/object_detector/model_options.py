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
"""Configurable model options for object detector models."""

import dataclasses


@dataclasses.dataclass
class ObjectDetectorModelOptions:
  """Configurable options for object detector model.

  Attributes:
    l2_weight_decay: L2 regularization penalty used in
      https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/L2.
  """

  l2_weight_decay: float = 3.0e-05
