# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""Configurable model options for face stylizer models."""

import dataclasses
from typing import List


# TODO: Add more detailed instructions about hyperparameter tuning.
@dataclasses.dataclass
class FaceStylizerModelOptions:
  """Configurable model options for face stylizer models.

  Attributes:
    swap_layers: The layers of feature to be interpolated between encoding
      features and StyleGAN input features.
    alpha: Weighting coefficient for swapping layer interpolation.
    adv_loss_weight: Weighting coeffcieint of adversarial loss versus perceptual
      loss.
  """

  swap_layers: List[int] = dataclasses.field(
      default_factory=lambda: [4, 5, 6, 7, 8, 9, 10, 11]
  )
  alpha: float = 1.0
  adv_loss_weight: float = 1.0
