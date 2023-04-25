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
"""Configurable model options for a BERT model."""

import dataclasses


@dataclasses.dataclass
class BertModelOptions:
  """Configurable model options for a BERT model.

  See https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding) for more details.

    Attributes:
      seq_len: Length of the sequence to feed into the model.
      do_fine_tuning: If true, then the BERT model is not frozen for training.
      dropout_rate: The rate for dropout.
  """
  seq_len: int = 128
  do_fine_tuning: bool = True
  dropout_rate: float = 0.1
