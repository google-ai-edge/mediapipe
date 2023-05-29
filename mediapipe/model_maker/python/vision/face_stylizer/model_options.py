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
"""Configurable model options for face stylizer models."""

from typing import Sequence
import dataclasses

from mediapipe.model_maker.python.core.utils import loss_functions


def _default_perceptual_quality_loss_weight():
  """Default perceptual quality loss weight for face stylizer."""
  return loss_functions.PerceptualLossWeight(l1=0.5, content=4.0, style=1.0)


# TODO: Add more detailed instructions about hyperparameter tuning.
@dataclasses.dataclass
class FaceStylizerModelOptions:
  """Configurable model options for face stylizer models.

  Attributes:
    swap_layers: The layers of feature to be interpolated between encoding
      features and StyleGAN input features.
    alpha: Weighting coefficient of style latent for swapping layer
      interpolation. Its valid range is [0, 1]. The greater weight means
      stronger style is applied to the output image. Expect to set it to a small
      value, i.e. < 0.1.
    perception_loss_weight: Weighting coefficients of image perception quality
      loss.
    adv_loss_weight: Weighting coeffcieint of adversarial loss versus image
      perceptual quality loss. It expects a small value, i.e. < 0.2.
  """

  swap_layers: Sequence[int] = dataclasses.field(
      default_factory=lambda: [4, 5, 10, 11]
  )
  alpha: float = 0.1
  perception_loss_weight: loss_functions.PerceptualLossWeight = (
      dataclasses.field(default_factory=_default_perceptual_quality_loss_weight)
  )
  adv_loss_weight: float = 0.2
