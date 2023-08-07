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
"""Specification for a BERT model."""

import dataclasses
from typing import Dict, Union

from mediapipe.model_maker.python.core import hyperparameters as hp
from mediapipe.model_maker.python.core.utils import file_util
from mediapipe.model_maker.python.text.core import bert_model_options

_DEFAULT_TFLITE_INPUT_NAME = {
    'ids': 'serving_default_input_word_ids:0',
    'mask': 'serving_default_input_mask:0',
    'segment_ids': 'serving_default_input_type_ids:0'
}


@dataclasses.dataclass
class BertModelSpec:
  """Specification for a BERT model.

  See https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding) for more details.

    Attributes:
      files: Either a TFHub url string which can be passed directly to
        hub.KerasLayer or a DownloadedFiles object of the model files.
      is_tf2: If True, the checkpoint is TF2 format. Else use TF1 format.
      hparams: Hyperparameters used for training.
      model_options: Configurable options for a BERT model.
      do_lower_case: boolean, whether to lower case the input text. Should be
        True / False for uncased / cased models respectively, where the models
        are specified by `downloaded_files`.
      tflite_input_name: Dict, input names for the TFLite model.
      name: The name of the object.
  """

  files: Union[str, file_util.DownloadedFiles]
  is_tf2: bool = True
  hparams: hp.BaseHParams = dataclasses.field(
      default_factory=lambda: hp.BaseHParams(
          epochs=3,
          batch_size=32,
          learning_rate=3e-5,
          distribution_strategy='mirrored',
      )
  )
  model_options: bert_model_options.BertModelOptions = dataclasses.field(
      default_factory=bert_model_options.BertModelOptions
  )
  do_lower_case: bool = True
  tflite_input_name: Dict[str, str] = dataclasses.field(
      default_factory=lambda: _DEFAULT_TFLITE_INPUT_NAME)
  name: str = 'Bert'

  def get_path(self) -> str:
    if isinstance(self.files, file_util.DownloadedFiles):
      return self.files.get_path()
    elif isinstance(self.files, str):
      return self.files
    else:
      raise ValueError(f'files has unsupported type: {type(self.files)}')
