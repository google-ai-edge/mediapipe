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
"""Specifications for text classifier models."""

import dataclasses
import enum
import functools

from mediapipe.model_maker.python.core import hyperparameters as hp
from mediapipe.model_maker.python.text.core import bert_model_spec
from mediapipe.model_maker.python.text.text_classifier import model_options as mo

# BERT-based text classifier spec inherited from BertModelSpec
BertClassifierSpec = bert_model_spec.BertModelSpec


@dataclasses.dataclass
class AverageWordEmbeddingClassifierSpec:
  """Specification for an average word embedding classifier model.

  Attributes:
    hparams: Configurable hyperparameters for training.
    model_options: Configurable options for the average word embedding model.
    name: The name of the object.
  """

  # `learning_rate` is unused for the average word embedding model
  hparams: hp.BaseHParams = hp.BaseHParams(
      epochs=10, batch_size=32, learning_rate=0)
  model_options: mo.AverageWordEmbeddingModelOptions = (
      mo.AverageWordEmbeddingModelOptions())
  name: str = 'AverageWordEmbedding'


average_word_embedding_classifier_spec = functools.partial(
    AverageWordEmbeddingClassifierSpec)

mobilebert_classifier_spec = functools.partial(
    BertClassifierSpec,
    hparams=hp.BaseHParams(
        epochs=3,
        batch_size=48,
        learning_rate=3e-5,
        distribution_strategy='off'),
    name='MobileBert',
    uri='https://tfhub.dev/tensorflow/mobilebert_en_uncased_L-24_H-128_B-512_A-4_F-4_OPT/1',
    tflite_input_name={
        'ids': 'serving_default_input_1:0',
        'mask': 'serving_default_input_3:0',
        'segment_ids': 'serving_default_input_2:0'
    },
)


@enum.unique
class SupportedModels(enum.Enum):
  """Predefined text classifier model specs supported by Model Maker."""
  AVERAGE_WORD_EMBEDDING_CLASSIFIER = average_word_embedding_classifier_spec
  MOBILEBERT_CLASSIFIER = mobilebert_classifier_spec
