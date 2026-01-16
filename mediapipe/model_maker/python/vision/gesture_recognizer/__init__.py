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
"""MediaPipe Model Maker Python Public API For Gesture Recognizer."""

from mediapipe.model_maker.python.vision.gesture_recognizer import dataset
from mediapipe.model_maker.python.vision.gesture_recognizer import gesture_recognizer
from mediapipe.model_maker.python.vision.gesture_recognizer import gesture_recognizer_options
from mediapipe.model_maker.python.vision.gesture_recognizer import hyperparameters
from mediapipe.model_maker.python.vision.gesture_recognizer import model_options

GestureRecognizer = gesture_recognizer.GestureRecognizer
ModelOptions = model_options.GestureRecognizerModelOptions
HParams = hyperparameters.HParams
Dataset = dataset.Dataset
HandDataPreprocessingParams = dataset.HandDataPreprocessingParams
GestureRecognizerOptions = gesture_recognizer_options.GestureRecognizerOptions

# Remove duplicated and non-public API
del constants  # pylint: disable=undefined-variable
del dataset
del gesture_recognizer
del gesture_recognizer_options
del hyperparameters
del metadata_writer  # pylint: disable=undefined-variable
del model_options
