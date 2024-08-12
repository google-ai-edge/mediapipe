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
import hashlib
import os
import re
from typing import Any, Mapping, Sequence, Union

import tensorflow as tf
import tensorflow_hub

from mediapipe.model_maker.python.core.data import cache_files as cache_files_lib
from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer
from mediapipe.model_maker.python.text.text_classifier import dataset as text_classifier_ds


def _validate_text_and_label(text: tf.Tensor, label: tf.Tensor) -> None:
  """Validates the shape and type of `text` and `label`.

  Args:
    text: Stores text data. Should have shape [1] and dtype tf.string.
    label: Stores the label for the corresponding `text`. Should have dtype
      tf.int64.

  Raises:
    ValueError: If either tensor has the wrong shape or type.
  """
  if text.shape != [1]:
    raise ValueError(f"`text` should have shape [1], got {text.shape}")
  if text.dtype != tf.string:
    raise ValueError(f"Expected dtype string for `text`, got {text.dtype}")
  if label.dtype != tf.int64:
    raise ValueError(f"Expected dtype int64 for `label`, got {label.dtype}")


def _decode_record(
    record: tf.Tensor, name_to_features: Mapping[str, tf.io.FixedLenFeature]
) -> Any:
  """Decodes a record into input for a BERT model.

  Args:
    record: Stores serialized example.
    name_to_features: Maps record keys to feature types.

  Returns:
    BERT model input features, label, and optional mask for the record.
  """
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  for name in list(example.keys()):
    example[name] = tf.cast(example[name], tf.int32)

  bert_features = {
      "input_word_ids": example["input_word_ids"],
      "input_mask": example["input_mask"],
      "input_type_ids": example["input_type_ids"],
  }
  if "label_mask" in example:
    return bert_features, example["label_ids"], example["label_mask"]
  else:
    return bert_features, example["label_ids"]


def _tfrecord_dataset(
    tfrecord_files: Sequence[str],
    name_to_features: Mapping[str, tf.io.FixedLenFeature],
) -> tf.data.TFRecordDataset:
  """Creates a single-file dataset to be passed for BERT custom training.

  Args:
    tfrecord_files: Filepaths for the dataset.
    name_to_features: Maps record keys to feature types.

  Returns:
    Dataset containing BERT model input features and labels.
  """
  d = tf.data.TFRecordDataset(tfrecord_files)
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
    model_name: Name of the model provided by the model_spec. Used to associate
      cached files with specific Bert model vocab.
    preprocessor: Which preprocessor to use. Must be one of the enum values of
      SupportedBertPreprocessors.
  """

  def __init__(
      self,
      seq_len: int,
      do_lower_case: bool,
      uri: str,
      model_name: str,
      tokenizer: bert_tokenizer.SupportedBertTokenizers,
  ):
    self._seq_len = seq_len
    # Vocab filepath is tied to the BERT module's URI.
    self._vocab_file = os.path.join(
        tensorflow_hub.resolve(uri), "assets", "vocab.txt"
    )
    self._do_lower_case = do_lower_case
    self._tokenizer: bert_tokenizer.BertTokenizer = None
    if tokenizer == bert_tokenizer.SupportedBertTokenizers.FULL_TOKENIZER:
      self._tokenizer = bert_tokenizer.BertFullTokenizer(
          self._vocab_file, self._do_lower_case, self._seq_len
      )
    elif (
        tokenizer == bert_tokenizer.SupportedBertTokenizers.FAST_BERT_TOKENIZER
    ):
      self._tokenizer = bert_tokenizer.BertFastTokenizer(
          self._vocab_file, self._do_lower_case, self._seq_len
      )
    else:
      raise ValueError(f"Unsupported tokenizer: {tokenizer}")
    self._model_name = model_name

  def _get_name_to_features(
      self, label_shape: int = 1, has_label_mask: bool = False
  ):
    """Gets the dictionary mapping record keys to feature types."""
    features = {
        "input_word_ids": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "input_mask": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "input_type_ids": tf.io.FixedLenFeature([self._seq_len], tf.int64),
        "label_ids": tf.io.FixedLenFeature([label_shape], tf.int64),
    }
    if has_label_mask:
      features["label_mask"] = tf.io.FixedLenFeature([label_shape], tf.int64)
    return features

  def get_vocab_file(self) -> str:
    """Returns the vocab file of the BertClassifierPreprocessor."""
    return self._vocab_file

  def get_tfrecord_cache_files(
      self, ds_cache_files
  ) -> cache_files_lib.TFRecordCacheFiles:
    """Helper to regenerate cache prefix filename using preprocessor info.

    We need to update the dataset cache_prefix cache because the actual cached
    dataset depends on the preprocessor parameters such as model_name, seq_len,
    and do_lower_case in addition to the raw dataset parameters which is already
    included in the ds_cache_files.cache_prefix_filename

    Specifically, the new cache_prefix_filename used by the preprocessor will
    be a hash generated from the following:
      1. cache_prefix_filename of the initial raw dataset
      2. model_name
      3. seq_len
      4. do_lower_case
      5. tokenizer name

    Args:
      ds_cache_files: TFRecordCacheFiles from the original raw dataset object

    Returns:
      A new TFRecordCacheFiles object which incorporates the preprocessor
      parameters.
    """
    hasher = hashlib.md5()
    hasher.update(ds_cache_files.cache_prefix_filename.encode("utf-8"))
    hasher.update(self._model_name.encode("utf-8"))
    hasher.update(str(self._seq_len).encode("utf-8"))
    hasher.update(str(self._do_lower_case).encode("utf-8"))
    hasher.update(self._tokenizer.name.encode("utf-8"))
    cache_prefix_filename = hasher.hexdigest()
    return cache_files_lib.TFRecordCacheFiles(
        cache_prefix_filename,
        ds_cache_files.cache_dir,
        ds_cache_files.num_shards,
    )

  def preprocess(
      self, dataset: text_classifier_ds.Dataset
  ) -> text_classifier_ds.Dataset:
    """Preprocesses data into input for a BERT-based classifier.

    Args:
      dataset: Stores (text, label) or (text, label, label_mask) data depending
        on whether dataset.has_label_mask is True.

    Returns:
      Dataset containing (bert_features, label) data.
    """
    ds_cache_files = dataset.tfrecord_cache_files
    # Get new tfrecord_cache_files by including preprocessor information.
    tfrecord_cache_files = self.get_tfrecord_cache_files(ds_cache_files)
    if not tfrecord_cache_files.is_cached():
      print(f"Writing new cache files to {tfrecord_cache_files.cache_prefix}")
      writers = tfrecord_cache_files.get_writers()
      size = 0
      for index, item in enumerate(dataset.gen_tf_dataset()):
        text, label = item[0], item[1]
        _validate_text_and_label(text, label)
        feature = self._tokenizer.process(text)
        def create_int_feature(values):
          f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
          return f

        features = collections.OrderedDict()
        features["input_word_ids"] = create_int_feature(
            feature["input_word_ids"]
        )
        features["input_mask"] = create_int_feature(feature["input_mask"])
        features["input_type_ids"] = create_int_feature(
            feature["input_type_ids"]
        )
        features["label_ids"] = create_int_feature(
            tf.reshape(label, [-1]).numpy().tolist()
        )
        if dataset.has_label_mask:
          mask = item[2]
          features["label_mask"] = create_int_feature(
              tf.reshape(mask, [-1]).numpy().tolist()
          )
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )
        writers[index % len(writers)].write(tf_example.SerializeToString())
        size = index + 1
      for writer in writers:
        writer.close()
      metadata = {"size": size, "label_names": dataset.label_names}
      tfrecord_cache_files.save_metadata(metadata)
    else:
      print(
          f"Using existing cache files at {tfrecord_cache_files.cache_prefix}"
      )
      metadata = tfrecord_cache_files.load_metadata()
    size = metadata["size"]
    label_names = metadata["label_names"]
    preprocessed_ds = _tfrecord_dataset(
        tfrecord_cache_files.tfrecord_files,
        self._get_name_to_features(dataset.label_shape, dataset.has_label_mask),
    )
    return text_classifier_ds.Dataset(
        dataset=preprocessed_ds,
        size=size,
        label_names=label_names,
        tfrecord_cache_files=tfrecord_cache_files,
        has_label_mask=dataset.has_label_mask,
        label_shape=dataset.label_shape,
    )

  @property
  def tokenizer(self) -> bert_tokenizer.BertTokenizer:
    return self._tokenizer


TextClassifierPreprocessor = Union[
    BertClassifierPreprocessor, AverageWordEmbeddingClassifierPreprocessor
]
