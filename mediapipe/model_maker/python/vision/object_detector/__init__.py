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
"""MediaPipe Model Maker Python Public API For Object Detector."""

from mediapipe.model_maker.python.vision.object_detector import dataset
from mediapipe.model_maker.python.vision.object_detector import hyperparameters
from mediapipe.model_maker.python.vision.object_detector import model_options
from mediapipe.model_maker.python.vision.object_detector import model_spec
from mediapipe.model_maker.python.vision.object_detector import object_detector
from mediapipe.model_maker.python.vision.object_detector import object_detector_options

ObjectDetector = object_detector.ObjectDetector
ModelOptions = model_options.ObjectDetectorModelOptions
ModelSpec = model_spec.ModelSpec
SupportedModels = model_spec.SupportedModels
HParams = hyperparameters.HParams
QATHParams = hyperparameters.QATHParams
Dataset = dataset.Dataset
ObjectDetectorOptions = object_detector_options.ObjectDetectorOptions

# Remove duplicated and non-public API
del dataset
del dataset_util  # pylint: disable=undefined-variable
del detection  # pylint: disable=undefined-variable
del hyperparameters
del model  # pylint: disable=undefined-variable
del model_options
del model_spec
del object_detector
del object_detector_options
del preprocessor  # pylint: disable=undefined-variable
