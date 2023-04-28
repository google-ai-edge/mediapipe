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
"""API for text classification."""

import abc
import os
import tempfile
from typing import Any, Optional, Sequence, Tuple

import tensorflow as tf
import tensorflow_hub as hub

from mediapipe.model_maker.python.core import hyperparameters as hp
from mediapipe.model_maker.python.core.data import dataset as ds
from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.text.text_classifier import dataset as text_ds
from mediapipe.model_maker.python.text.text_classifier import model_options as mo
from mediapipe.model_maker.python.text.text_classifier import model_spec as ms
from mediapipe.model_maker.python.text.text_classifier import preprocessor
from mediapipe.model_maker.python.text.text_classifier import text_classifier_options
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import text_classifier as text_classifier_writer


def _validate(options: text_classifier_options.TextClassifierOptions):
  """Validates that `model_options` and `supported_model` are compatible.

  Args:
    options: Options for creating and training a text classifier.

  Raises:
    ValueError if there is a mismatch between `model_options` and
    `supported_model`.
  """

  if options.model_options is None:
    return

  if (isinstance(options.model_options, mo.AverageWordEmbeddingModelOptions) and
      (options.supported_model !=
       ms.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER)):
    raise ValueError("Expected AVERAGE_WORD_EMBEDDING_CLASSIFIER,"
                     f" got {options.supported_model}")
  if (isinstance(options.model_options, mo.BertModelOptions) and
      (options.supported_model != ms.SupportedModels.MOBILEBERT_CLASSIFIER)):
    raise ValueError(
        f"Expected MOBILEBERT_CLASSIFIER, got {options.supported_model}")


class TextClassifier(classifier.Classifier):
  """API for creating and training a text classification model."""

  def __init__(self, model_spec: Any, hparams: hp.BaseHParams,
               label_names: Sequence[str]):
    super().__init__(
        model_spec=model_spec, label_names=label_names, shuffle=hparams.shuffle)
    self._model_spec = model_spec
    self._hparams = hparams
    self._callbacks = model_util.get_default_callbacks(self._hparams.export_dir)
    self._text_preprocessor: preprocessor.TextClassifierPreprocessor = None

  @classmethod
  def create(
      cls, train_data: text_ds.Dataset, validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions
  ) -> "TextClassifier":
    """Factory function that creates and trains a text classifier.

    Note that `train_data` and `validation_data` are expected to share the same
    `label_names` since they should be split from the same dataset.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Options for creating and training the text classifier.

    Returns:
      A text classifier.

    Raises:
      ValueError if `train_data` and `validation_data` do not have the
      same label_names or `options` contains an unknown `supported_model`
    """
    if train_data.label_names != validation_data.label_names:
      raise ValueError(
          f"Training data label names {train_data.label_names} not equal to "
          f"validation data label names {validation_data.label_names}")

    _validate(options)
    if options.model_options is None:
      options.model_options = options.supported_model.value().model_options

    if options.hparams is None:
      options.hparams = options.supported_model.value().hparams

    if options.supported_model == ms.SupportedModels.MOBILEBERT_CLASSIFIER:
      text_classifier = (
          _BertClassifier.create_bert_classifier(train_data, validation_data,
                                                 options,
                                                 train_data.label_names))
    elif (options.supported_model ==
          ms.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER):
      text_classifier = (
          _AverageWordEmbeddingClassifier
          .create_average_word_embedding_classifier(train_data, validation_data,
                                                    options,
                                                    train_data.label_names))
    else:
      raise ValueError(f"Unknown model {options.supported_model}")

    return text_classifier

  def evaluate(self, data: ds.Dataset, batch_size: int = 32) -> Any:
    """Overrides Classifier.evaluate().

    Args:
      data: Evaluation dataset. Must be a TextClassifier Dataset.
      batch_size: Number of samples per evaluation step.

    Returns:
      The loss value and accuracy.

    Raises:
      ValueError if `data` is not a TextClassifier Dataset.
    """
    # This override is needed because TextClassifier preprocesses its data
    # outside of the `gen_tf_dataset()` method. The preprocess call also
    # requires a TextClassifier Dataset instead of a core Dataset.
    if not isinstance(data, text_ds.Dataset):
      raise ValueError("Need a TextClassifier Dataset.")

    processed_data = self._text_preprocessor.preprocess(data)
    dataset = processed_data.gen_tf_dataset(batch_size, is_training=False)
    return self._model.evaluate(dataset)

  def export_model(
      self,
      model_name: str = "model.tflite",
      quantization_config: Optional[quantization.QuantizationConfig] = None):
    """Converts and saves the model to a TFLite file with metadata included.

    Note that only the TFLite file is needed for deployment. This function also
    saves a metadata.json file to the same directory as the TFLite file which
    can be used to interpret the metadata content in the TFLite file.

    Args:
      model_name: File name to save TFLite model with metadata. The full export
        path is {self._hparams.export_dir}/{model_name}.
      quantization_config: The configuration for model quantization.
    """
    if not tf.io.gfile.exists(self._hparams.export_dir):
      tf.io.gfile.makedirs(self._hparams.export_dir)
    tflite_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, "metadata.json")

    tflite_model = model_util.convert_to_tflite(
        model=self._model, quantization_config=quantization_config)
    vocab_filepath = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    self._save_vocab(vocab_filepath)

    writer = self._get_metadata_writer(tflite_model, vocab_filepath)
    tflite_model_with_metadata, metadata_json = writer.populate()
    model_util.save_tflite(tflite_model_with_metadata, tflite_file)
    with open(metadata_file, "w") as f:
      f.write(metadata_json)

  @abc.abstractmethod
  def _save_vocab(self, vocab_filepath: str):
    """Saves the preprocessor's vocab to `vocab_filepath`."""

  @abc.abstractmethod
  def _get_metadata_writer(self, tflite_model: bytearray, vocab_filepath: str):
    """Gets the metadata writer for the text classifier TFLite model."""


class _AverageWordEmbeddingClassifier(TextClassifier):
  """APIs to help create and train an Average Word Embedding text classifier."""

  _DELIM_REGEX_PATTERN = r"[^\w\']+"

  def __init__(self, model_spec: ms.AverageWordEmbeddingClassifierSpec,
               model_options: mo.AverageWordEmbeddingModelOptions,
               hparams: hp.BaseHParams, label_names: Sequence[str]):
    super().__init__(model_spec, hparams, label_names)
    self._model_options = model_options
    self._loss_function = "sparse_categorical_crossentropy"
    self._metric_function = "accuracy"
    self._text_preprocessor: (
        preprocessor.AverageWordEmbeddingClassifierPreprocessor) = None

  @classmethod
  def create_average_word_embedding_classifier(
      cls, train_data: text_ds.Dataset, validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions,
      label_names: Sequence[str]) -> "_AverageWordEmbeddingClassifier":
    """Creates, trains, and returns an Average Word Embedding classifier.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Options for creating and training the text classifier.
      label_names: Label names used in the data.

    Returns:
      An Average Word Embedding classifier.
    """
    average_word_embedding_classifier = _AverageWordEmbeddingClassifier(
        model_spec=options.supported_model.value(),
        model_options=options.model_options,
        hparams=options.hparams,
        label_names=train_data.label_names)
    average_word_embedding_classifier._create_and_train_model(
        train_data, validation_data)
    return average_word_embedding_classifier

  def _create_and_train_model(self, train_data: text_ds.Dataset,
                              validation_data: text_ds.Dataset):
    """Creates the Average Word Embedding classifier keras model and trains it.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    (processed_train_data, processed_validation_data) = (
        self._load_and_run_preprocessor(train_data, validation_data))
    self._create_model()
    self._optimizer = "rmsprop"
    self._train_model(processed_train_data, processed_validation_data)

  def _load_and_run_preprocessor(
      self, train_data: text_ds.Dataset, validation_data: text_ds.Dataset
  ) -> Tuple[text_ds.Dataset, text_ds.Dataset]:
    """Runs an AverageWordEmbeddingClassifierPreprocessor on the data.

    Args:
      train_data: Training data.
      validation_data: Validation data.

    Returns:
      Preprocessed training data and preprocessed validation data.
    """
    train_texts = [text.numpy()[0] for text, _ in train_data.gen_tf_dataset()]
    validation_texts = [
        text.numpy()[0] for text, _ in validation_data.gen_tf_dataset()
    ]
    self._text_preprocessor = (
        preprocessor.AverageWordEmbeddingClassifierPreprocessor(
            seq_len=self._model_options.seq_len,
            do_lower_case=self._model_options.do_lower_case,
            texts=train_texts + validation_texts,
            vocab_size=self._model_options.vocab_size))
    return self._text_preprocessor.preprocess(
        train_data), self._text_preprocessor.preprocess(validation_data)

  def _create_model(self):
    """Creates an Average Word Embedding model."""
    self._model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            input_shape=[self._model_options.seq_len],
            dtype=tf.int32,
            name="input_ids",
        ),
        tf.keras.layers.Embedding(
            len(self._text_preprocessor.get_vocab()),
            self._model_options.wordvec_dim,
            input_length=self._model_options.seq_len,
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(
            self._model_options.wordvec_dim, activation=tf.nn.relu
        ),
        tf.keras.layers.Dropout(self._model_options.dropout_rate),
        tf.keras.layers.Dense(self._num_classes, activation="softmax"),
    ])

  def _save_vocab(self, vocab_filepath: str):
    with tf.io.gfile.GFile(vocab_filepath, "w") as f:
      for token, index in self._text_preprocessor.get_vocab().items():
        f.write(f"{token} {index}\n")

  def _get_metadata_writer(self, tflite_model: bytearray, vocab_filepath: str):
    return text_classifier_writer.MetadataWriter.create_for_regex_model(
        model_buffer=tflite_model,
        regex_tokenizer=metadata_writer.RegexTokenizer(
            # TODO: Align with MediaPipe's RegexTokenizer.
            delim_regex_pattern=self._DELIM_REGEX_PATTERN,
            vocab_file_path=vocab_filepath),
        labels=metadata_writer.Labels().add(list(self._label_names)))


class _BertClassifier(TextClassifier):
  """APIs to help create and train a BERT-based text classifier."""

  _INITIALIZER_RANGE = 0.02

  def __init__(self, model_spec: ms.BertClassifierSpec,
               model_options: mo.BertModelOptions, hparams: hp.BaseHParams,
               label_names: Sequence[str]):
    super().__init__(model_spec, hparams, label_names)
    self._model_options = model_options
    self._loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    self._metric_function = tf.keras.metrics.SparseCategoricalAccuracy(
        "test_accuracy", dtype=tf.float32)
    self._text_preprocessor: preprocessor.BertClassifierPreprocessor = None

  @classmethod
  def create_bert_classifier(
      cls, train_data: text_ds.Dataset, validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions,
      label_names: Sequence[str]) -> "_BertClassifier":
    """Creates, trains, and returns a BERT-based classifier.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Options for creating and training the text classifier.
      label_names: Label names used in the data.

    Returns:
      A BERT-based classifier.
    """
    bert_classifier = _BertClassifier(
        model_spec=options.supported_model.value(),
        model_options=options.model_options,
        hparams=options.hparams,
        label_names=train_data.label_names)
    bert_classifier._create_and_train_model(train_data, validation_data)
    return bert_classifier

  def _create_and_train_model(self, train_data: text_ds.Dataset,
                              validation_data: text_ds.Dataset):
    """Creates the BERT-based classifier keras model and trains it.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    (processed_train_data, processed_validation_data) = (
        self._load_and_run_preprocessor(train_data, validation_data))
    self._create_model()
    self._create_optimizer(processed_train_data)
    self._train_model(processed_train_data, processed_validation_data)

  def _load_and_run_preprocessor(
      self, train_data: text_ds.Dataset, validation_data: text_ds.Dataset
  ) -> Tuple[text_ds.Dataset, text_ds.Dataset]:
    """Loads a BertClassifierPreprocessor and runs it on the data.

    Args:
      train_data: Training data.
      validation_data: Validation data.

    Returns:
      Preprocessed training data and preprocessed validation data.
    """
    self._text_preprocessor = preprocessor.BertClassifierPreprocessor(
        seq_len=self._model_options.seq_len,
        do_lower_case=self._model_spec.do_lower_case,
        uri=self._model_spec.downloaded_files.get_path(),
    )
    return (self._text_preprocessor.preprocess(train_data),
            self._text_preprocessor.preprocess(validation_data))

  def _create_model(self):
    """Creates a BERT-based classifier model.

    The model architecture consists of stacking a dense classification layer and
    dropout layer on top of the BERT encoder outputs.
    """
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,), dtype=tf.int32),
    )
    encoder = hub.KerasLayer(
        self._model_spec.downloaded_files.get_path(),
        trainable=self._model_options.do_fine_tuning,
    )
    encoder_outputs = encoder(encoder_inputs)
    pooled_output = encoder_outputs["pooled_output"]

    output = tf.keras.layers.Dropout(rate=self._model_options.dropout_rate)(
        pooled_output)
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=self._INITIALIZER_RANGE)
    output = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=initializer,
        name="output",
        activation="softmax",
        dtype=tf.float32)(
            output)
    self._model = tf.keras.Model(inputs=encoder_inputs, outputs=output)

  def _create_optimizer(self, train_data: text_ds.Dataset):
    """Loads an optimizer with a learning rate schedule.

    The decay steps in the learning rate schedule depend on the
    `steps_per_epoch` which may depend on the size of the training data.

    Args:
      train_data: Training data.
    """
    self._hparams.steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=self._hparams.steps_per_epoch,
        batch_size=self._hparams.batch_size,
        train_data=train_data)
    total_steps = self._hparams.steps_per_epoch * self._hparams.epochs
    warmup_steps = int(total_steps * 0.1)
    initial_lr = self._hparams.learning_rate
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        end_learning_rate=0.0,
        power=1.0)
    if warmup_steps:
      lr_schedule = model_util.WarmUp(
          initial_learning_rate=initial_lr,
          decay_schedule_fn=lr_schedule,
          warmup_steps=warmup_steps)

    self._optimizer = tf.keras.optimizers.experimental.AdamW(
        lr_schedule, weight_decay=0.01, epsilon=1e-6, global_clipnorm=1.0)
    self._optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"])

  def _save_vocab(self, vocab_filepath: str):
    tf.io.gfile.copy(
        self._text_preprocessor.get_vocab_file(),
        vocab_filepath,
        overwrite=True)

  def _get_metadata_writer(self, tflite_model: bytearray, vocab_filepath: str):
    return text_classifier_writer.MetadataWriter.create_for_bert_model(
        model_buffer=tflite_model,
        tokenizer=metadata_writer.BertTokenizer(vocab_filepath),
        labels=metadata_writer.Labels().add(list(self._label_names)),
        ids_name=self._model_spec.tflite_input_name["ids"],
        mask_name=self._model_spec.tflite_input_name["mask"],
        segment_name=self._model_spec.tflite_input_name["segment_ids"])
