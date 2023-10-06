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
"""Options for building object detector."""

import dataclasses
from typing import Optional

from mediapipe.model_maker.python.vision.object_detector import hyperparameters
from mediapipe.model_maker.python.vision.object_detector import model_options as model_opt
from mediapipe.model_maker.python.vision.object_detector import model_spec


@dataclasses.dataclass
class ObjectDetectorOptions:
  """Configurable options for building object detector.

  Attributes:
    supported_model: A model from the SupportedModels enum.
    model_options: A set of options for configuring the selected model.
    hparams: A set of hyperparameters used to train the object detector.
  """

  supported_model: model_spec.SupportedModels
  model_options: Optional[model_opt.ObjectDetectorModelOptions] = None
  hparams: Optional[hyperparameters.HParams] = None
