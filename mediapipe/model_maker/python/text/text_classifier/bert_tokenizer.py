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
"""Text classifier BERT tokenizer library."""
import abc
import enum
from typing import Mapping, Sequence

import tensorflow as tf
import tensorflow_text as tf_text

from official.nlp.tools import tokenization


@enum.unique
class SupportedBertTokenizers(enum.Enum):
  """Supported preprocessors."""

  FULL_TOKENIZER = "fulltokenizer"
  FAST_BERT_TOKENIZER = "fastberttokenizer"


class BertTokenizer(abc.ABC):
  """Abstract BertTokenizer class."""

  name: str

  @abc.abstractmethod
  def __init__(self, vocab_file: str, do_lower_case: bool, seq_len: int):
    pass

  @abc.abstractmethod
  def process(self, input_tensor: tf.Tensor) -> Mapping[str, Sequence[int]]:
    pass


class BertFullTokenizer(BertTokenizer):
  """Tokenizer using the FullTokenizer from tensorflow_models."""

  name = "fulltokenizer"

  def __init__(self, vocab_file: str, do_lower_case: bool, seq_len: int):
    self._tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )
    self._seq_len = seq_len

  def process(self, input_tensor: tf.Tensor) -> Mapping[str, Sequence[int]]:
    """Processes one input_tensor example.

    Args:
      input_tensor: A tensor with shape (1, None) of a utf-8 encoded string.

    Returns:
      A dictionary of lists all with shape (1, self._seq_len) containing the
        keys "input_word_ids", "input_type_ids", and "input_mask".
    """
    tokens = self._tokenizer.tokenize(input_tensor.numpy()[0].decode("utf-8"))
    tokens = tokens[0 : (self._seq_len - 2)]  # account for [CLS] and [SEP]
    tokens.insert(0, "[CLS]")
    tokens.append("[SEP]")
    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self._seq_len:
      input_ids.append(0)
      input_mask.append(0)
    segment_ids = [0] * self._seq_len
    return {
        "input_word_ids": input_ids,
        "input_type_ids": segment_ids,
        "input_mask": input_mask,
    }


class BertFastTokenizer(BertTokenizer):
  """Tokenizer using the FastBertTokenizer from tensorflow_text.

  For more information, see:
  https://www.tensorflow.org/text/api_docs/python/text/FastBertTokenizer
  """

  name = "fastberttokenizer"

  def __init__(self, vocab_file: str, do_lower_case: bool, seq_len: int):
    with tf.io.gfile.GFile(vocab_file, "r") as f:
      vocab = f.read().splitlines()
    self._tokenizer = tf_text.FastBertTokenizer(
        vocab=vocab,
        token_out_type=tf.int32,
        support_detokenization=False,
        lower_case_nfd_strip_accents=do_lower_case,
    )
    self._seq_len = seq_len
    self._cls_id = vocab.index("[CLS]")
    self._sep_id = vocab.index("[SEP]")
    self._pad_id = vocab.index("[PAD]")

  def process_fn(self, input_tensor: tf.Tensor) -> Mapping[str, tf.Tensor]:
    """Tensor implementation of the process function.

    This implementation can be used within a model graph directly since it
    takes in tensors and outputs tensors.

    Args:
      input_tensor: Input string tensor

    Returns:
      Dictionary of tf.Tensors.
    """
    input_ids = self._tokenizer.tokenize(input_tensor).flat_values
    input_ids = input_ids[: (self._seq_len - 2)]
    input_ids = tf.concat(
        [
            tf.constant([self._cls_id]),
            input_ids,
            tf.constant([self._sep_id]),
            tf.fill((self._seq_len,), self._pad_id),
        ],
        axis=0,
    )
    input_ids = input_ids[: self._seq_len]
    input_type_ids = tf.zeros(self._seq_len, dtype=tf.int32)
    input_mask = tf.cast(input_ids != self._pad_id, dtype=tf.int32)
    return {
        "input_word_ids": input_ids,
        "input_type_ids": input_type_ids,
        "input_mask": input_mask,
    }

  def process(self, input_tensor: tf.Tensor) -> Mapping[str, Sequence[int]]:
    """Processes one input_tensor example.

    Args:
      input_tensor: A tensor with shape (1, None) of a utf-8 encoded string.

    Returns:
      A dictionary of lists all with shape (1, self._seq_len) containing the
        keys "input_word_ids", "input_type_ids", and "input_mask".
    """
    result = self.process_fn(input_tensor)
    return {k: v.numpy().tolist() for k, v in result.items()}
