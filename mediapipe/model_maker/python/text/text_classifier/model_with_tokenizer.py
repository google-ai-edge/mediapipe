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
"""Text classifier export module library."""
import tensorflow as tf


class ModelWithTokenizer(tf.keras.Model):
  """A model with the tokenizer included in graph for exporting to TFLite."""

  def __init__(self, tokenizer, model):
    super().__init__()
    self._tokenizer = tokenizer
    self._model = model

  @tf.function(
      input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.string, name="input")
      ]
  )
  def call(self, input_tensor):
    x = self._tokenizer.process_fn(input_tensor)
    x = {k: tf.expand_dims(v, axis=0) for k, v in x.items()}
    x = self._model(x)
    return x[0]  # TODO: Add back the batch dimension
