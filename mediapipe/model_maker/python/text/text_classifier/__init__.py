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
"""MediaPipe Public Python API for Text Classifier."""

from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer
from mediapipe.model_maker.python.text.text_classifier import dataset
from mediapipe.model_maker.python.text.text_classifier import hyperparameters
from mediapipe.model_maker.python.text.text_classifier import model_options
from mediapipe.model_maker.python.text.text_classifier import model_spec
from mediapipe.model_maker.python.text.text_classifier import preprocessor
from mediapipe.model_maker.python.text.text_classifier import text_classifier
from mediapipe.model_maker.python.text.text_classifier import text_classifier_options


AverageWordEmbeddingHParams = hyperparameters.AverageWordEmbeddingHParams
AverageWordEmbeddingModelOptions = (
    model_options.AverageWordEmbeddingModelOptions
)
BertOptimizer = hyperparameters.BertOptimizer
BertHParams = hyperparameters.BertHParams
BertModelOptions = model_options.BertModelOptions
CSVParams = dataset.CSVParameters
Dataset = dataset.Dataset
SupportedModels = model_spec.SupportedModels
TextClassifier = text_classifier.TextClassifier
TextClassifierOptions = text_classifier_options.TextClassifierOptions
SupportedBertTokenizers = bert_tokenizer.SupportedBertTokenizers

# Remove duplicated and non-public API
del bert_tokenizer
del hyperparameters
del dataset
del model_options
del model_spec
del preprocessor
del text_classifier
del text_classifier_options
