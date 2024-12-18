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
"""Specifications for text classifier models."""

import dataclasses
import enum
import functools

from mediapipe.model_maker.python.text.core import bert_model_spec
from mediapipe.model_maker.python.text.text_classifier import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import model_options as mo


MOBILEBERT_FILES = (
    'https://tfhub.dev/google/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT/1'
)


@dataclasses.dataclass
class AverageWordEmbeddingClassifierSpec:
  """Specification for an average word embedding classifier model.

  Attributes:
    hparams: Configurable hyperparameters for training.
    model_options: Configurable options for the average word embedding model.
    name: The name of the object.
  """

  # `learning_rate` is unused for the average word embedding model
  hparams: hp.AverageWordEmbeddingHParams = dataclasses.field(
      default_factory=lambda: hp.AverageWordEmbeddingHParams(
          epochs=10, batch_size=32, learning_rate=0
      )
  )
  model_options: mo.AverageWordEmbeddingModelOptions = dataclasses.field(
      default_factory=mo.AverageWordEmbeddingModelOptions
  )
  name: str = 'AverageWordEmbedding'


average_word_embedding_classifier_spec = functools.partial(
    AverageWordEmbeddingClassifierSpec
)


@dataclasses.dataclass
class BertClassifierSpec(bert_model_spec.BertModelSpec):
  """Specification for a Bert classifier model.

  Only overrides the hparams attribute since the rest of the attributes are
  inherited from the BertModelSpec.
  """

  hparams: hp.BertHParams = dataclasses.field(default_factory=hp.BertHParams)


mobilebert_classifier_spec = functools.partial(
    BertClassifierSpec,
    files=MOBILEBERT_FILES,
    hparams=hp.BertHParams(
        epochs=3, batch_size=48, learning_rate=3e-5, distribution_strategy='off'
    ),
    name='MobileBERT',
    is_tf2=False,
)


@enum.unique
class SupportedModels(enum.Enum):
  """Predefined text classifier model specs supported by Model Maker."""

  AVERAGE_WORD_EMBEDDING_CLASSIFIER = average_word_embedding_classifier_spec
  MOBILEBERT_CLASSIFIER = mobilebert_classifier_spec
