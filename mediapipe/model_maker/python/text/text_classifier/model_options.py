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
"""Configurable model options for text classifier models."""

import dataclasses
from typing import Union

from mediapipe.model_maker.python.text.core import bert_model_options

# BERT text classifier model options inherited from BertModelOptions.
BertModelOptions = bert_model_options.BertModelOptions


@dataclasses.dataclass
class AverageWordEmbeddingModelOptions:
  """Configurable model options for an Average Word Embedding classifier.

  Attributes:
    seq_len: Length of the sequence to feed into the model.
    wordvec_dim: Dimension of the word embedding.
    do_lower_case: Whether to convert all uppercase characters to lowercase
      during preprocessing.
    vocab_size: Number of words to generate the vocabulary from data.
    dropout_rate: The rate for dropout.
  """
  seq_len: int = 256
  wordvec_dim: int = 16
  do_lower_case: bool = True
  vocab_size: int = 10000
  dropout_rate: float = 0.2


TextClassifierModelOptions = Union[AverageWordEmbeddingModelOptions,
                                   BertModelOptions]
