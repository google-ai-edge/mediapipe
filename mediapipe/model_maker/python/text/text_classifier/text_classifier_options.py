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
"""User-facing customization options to create and train a text classifier."""

import dataclasses
from typing import Optional

from mediapipe.model_maker.python.text.text_classifier import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import model_options as mo
from mediapipe.model_maker.python.text.text_classifier import model_spec as ms


@dataclasses.dataclass
class TextClassifierOptions:
  """User-facing options for creating the text classifier.

  Attributes:
    supported_model: A preconfigured model spec.
    hparams: Training hyperparameters the user can set to override the ones in
      `supported_model`.
    model_options: Model options the user can set to override the ones in
      `supported_model`. The model options type should be consistent with the
      architecture of the `supported_model`.
  """
  supported_model: ms.SupportedModels
  hparams: Optional[hp.HParams] = None
  model_options: Optional[mo.TextClassifierModelOptions] = None
