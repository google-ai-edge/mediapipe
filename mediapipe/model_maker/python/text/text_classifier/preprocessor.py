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
# ==============================================================================
"""Preprocessors for text classification."""

import collections
import os
import re
import tempfile
from typing import Mapping, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_hub

from mediapipe.model_maker.python.text.text_classifier import dataset as text_classifier_ds
from official.nlp.data import classifier_data_lib
from official.nlp.tools import tokenization


def _validate_text_and_label(text: tf.Tensor, label: tf.Tensor) -> None:
  """Validates the shape and type of `text` and `label`.

  Args:
    text: Stores text data. Should have shape [1] and dtype tf.string.
    label: Stores the label for the corresponding `text`. Should have shape [1]
      and dtype tf.int64.

  Raises:
    ValueError: If either tensor has the wrong shape or type.
  """
  if text.shape != [1]:
    raise ValueError(f"`text` should have shape [1], got {text.shape}")
  if text.dtype != tf.string:
    raise ValueError(f"Expected dtype string for `text`, got {text.dtype}")
  if label.shape != [1]:
    raise ValueError(f"`label` should have shape [1], got {text.shape}")
  if label.dtype != tf.int64:
    raise ValueError(f"Expected dtype int64 for `label`, got {label.dtype}")


def _decode_record(
    record: tf.Tensor, name_to_features: Mapping[str, tf.io.FixedLenFeature]
) -> Tuple[Mapping[str, tf.Tensor], tf.Tensor]:
  """Decodes a record into input for a BERT model.

  Args:
    record: Stores serialized example.
    name_to_features: Maps record keys to feature types.

  Returns:
    BERT model input features and label for the record.
  """
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  for name in list(example.keys()):
    example[name] = tf.cast(example[name], tf.int32)

  bert_features = {
      "input_word_ids": example["input_ids"],
      "input_mask": example["input_mask"],
      "input_type_ids": example["segment_ids"]
  }
  return bert_features, example["label_ids"]


def _single_file_dataset(
    input_file: str, name_to_features: Mapping[str, tf.io.FixedLenFeature]
) -> tf.data.TFRecordDataset:
  """Creates a single-file dataset to be passed for BERT custom training.

  Args:
    input_file: Filepath for the dataset.
    name_to_features: Maps record keys to feature types.

  Returns:
    Dataset containing BERT model input features and labels.
  """
  d = tf.data.TFRecordDataset(input_file)
  d = d.map(
      lambda record: _decode_record(record, name_to_features),
      num_parallel_calls=tf.data.AUTOTUNE)
  return d


class AverageWordEmbeddingClassifierPreprocessor:
  """Preprocessor for an Average Word Embedding model.

  Takes (text, label) data and applies regex tokenization and padding to the
  text to generate (token IDs, label) data.

  Attributes:
    seq_len: Length of the input sequence to the model.
    do_lower_case: Whether text inputs should be converted to lower-case.
    vocab: Vocabulary of tokens used by the model.
  """

  PAD: str = "<PAD>"  # Index: 0
  START: str = "<START>"  # Index: 1
  UNKNOWN: str = "<UNKNOWN>"  # Index: 2

  def __init__(self, seq_len: int, do_lower_case: bool, texts: Sequence[str],
               vocab_size: int):
    self._seq_len = seq_len
    self._do_lower_case = do_lower_case
    self._vocab = self._gen_vocab(texts, vocab_size)

  def _gen_vocab(self, texts: Sequence[str],
                 vocab_size: int) -> Mapping[str, int]:
    """Generates vocabulary list in `texts` with size `vocab_size`.

    Args:
      texts: All texts (across training and validation data) that will be
        preprocessed by the model.
      vocab_size: Size of the vocab.

    Returns:
      The vocab mapping tokens to IDs.
    """
    vocab_counter = collections.Counter()

    for text in texts:
      tokens = self._regex_tokenize(text)
      for token in tokens:
        vocab_counter[token] += 1

    vocab_freq = vocab_counter.most_common(vocab_size)
    vocab_list = [self.PAD, self.START, self.UNKNOWN
                 ] + [word for word, _ in vocab_freq]
    return collections.OrderedDict(((v, i) for i, v in enumerate(vocab_list)))

  def get_vocab(self) -> Mapping[str, int]:
    """Returns the vocab of the AverageWordEmbeddingClassifierPreprocessor."""
    return self._vocab

  # TODO: Align with MediaPipe's RegexTokenizer.
  def _regex_tokenize(self, text: str) -> Sequence[str]:
    """Splits `text` by words but does not split on single quotes.

    Args:
      text: Text to be tokenized.

    Returns:
      List of tokens.
    """
    text = tf.compat.as_text(text)
    if self._do_lower_case:
      text = text.lower()
    tokens = re.compile(r"[^\w\']+").split(text.strip())
    # Filters out any empty strings in `tokens`.
    return list(filter(None, tokens))

  def _tokenize_and_pad(self, text: str) -> Sequence[int]:
    """Tokenizes `text` and pads the tokens to `seq_len`.

    Args:
      text: Text to be tokenized and padded.

    Returns:
      List of token IDs padded to have length `seq_len`.
    """
    tokens = self._regex_tokenize(text)

    # Gets ids for START, PAD and UNKNOWN tokens.
    start_id = self._vocab[self.START]
    pad_id = self._vocab[self.PAD]
    unknown_id = self._vocab[self.UNKNOWN]

    token_ids = [self._vocab.get(token, unknown_id) for token in tokens]
    token_ids = [start_id] + token_ids

    if len(token_ids) < self._seq_len:
      pad_length = self._seq_len - len(token_ids)
      token_ids = token_ids + pad_length * [pad_id]
    else:
      token_ids = token_ids[:self._seq_len]
    return token_ids

  def preprocess(
      self, dataset: text_classifier_ds.Dataset) -> text_classifier_ds.Dataset:
    """Preprocesses data into input for an Average Word Embedding model.

    Args:
      dataset: Stores (text, label) data.

    Returns:
      Dataset containing (token IDs, label) data.
    """
    token_ids_list = []
    labels_list = []
    for text, label in dataset.gen_tf_dataset():
      _validate_text_and_label(text, label)
      token_ids = self._tokenize_and_pad(text.numpy()[0].decode("utf-8"))
      token_ids_list.append(token_ids)
      labels_list.append(label.numpy()[0])

    token_ids_ds = tf.data.Dataset.from_tensor_slices(token_ids_list)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels_list)
    preprocessed_ds = tf.data.Dataset.zip((token_ids_ds, labels_ds))
    return text_classifier_ds.Dataset(
        dataset=preprocessed_ds,
        size=dataset.size,
        label_names=dataset.label_names)


class BertClassifierPreprocessor:
  """Preprocessor for a BERT-based classifier.

  Attributes:
    seq_len: Length of the input sequence to the model.
    vocab_file: File containing the BERT vocab.
    tokenizer: BERT tokenizer.
  """

  def __init__(self, seq_len: int, do_lower_case: bool, uri: str):
    self._seq_len = seq_len
    # Vocab filepath is tied to the BERT module's URI.
    self._vocab_file = os.path.join(
        tensorflow_hub.resolve(uri), "assets", "vocab.txt")
    self._tokenizer = tokenization.FullTokenizer(self._vocab_file,
                                                 do_lower_case)

  def _get_name_to_features(self):
    """Gets the dictionary mapping record keys to feature types."""
    return {
        "input_ids": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "input_mask": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "label_ids": tf.io.FixedLenFeature([], tf.int64),
    }

  def get_vocab_file(self) -> str:
    """Returns the vocab file of the BertClassifierPreprocessor."""
    return self._vocab_file

  def preprocess(
      self, dataset: text_classifier_ds.Dataset) -> text_classifier_ds.Dataset:
    """Preprocesses data into input for a BERT-based classifier.

    Args:
      dataset: Stores (text, label) data.

    Returns:
      Dataset containing (bert_features, label) data.
    """
    examples = []
    for index, (text, label) in enumerate(dataset.gen_tf_dataset()):
      _validate_text_and_label(text, label)
      examples.append(
          classifier_data_lib.InputExample(
              guid=str(index),
              text_a=text.numpy()[0].decode("utf-8"),
              text_b=None,
              # InputExample expects the label name rather than the int ID
              label=dataset.label_names[label.numpy()[0]]))

    tfrecord_file = os.path.join(tempfile.mkdtemp(), "bert_features.tfrecord")
    classifier_data_lib.file_based_convert_examples_to_features(
        examples=examples,
        label_list=dataset.label_names,
        max_seq_length=self._seq_len,
        tokenizer=self._tokenizer,
        output_file=tfrecord_file)
    preprocessed_ds = _single_file_dataset(tfrecord_file,
                                           self._get_name_to_features())
    return text_classifier_ds.Dataset(
        dataset=preprocessed_ds,
        size=dataset.size,
        label_names=dataset.label_names)


TextClassifierPreprocessor = (
    Union[BertClassifierPreprocessor,
          AverageWordEmbeddingClassifierPreprocessor])
