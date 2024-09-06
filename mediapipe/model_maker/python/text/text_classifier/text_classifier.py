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
from tensorflow_addons import optimizers as tfa_optimizers
import tensorflow_hub as hub

from mediapipe.model_maker.python.core.data import dataset as ds
from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import hub_loader
from mediapipe.model_maker.python.core.utils import loss_functions
from mediapipe.model_maker.python.core.utils import metrics
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.text.text_classifier import bert_tokenizer
from mediapipe.model_maker.python.text.text_classifier import dataset as text_ds
from mediapipe.model_maker.python.text.text_classifier import hyperparameters as hp
from mediapipe.model_maker.python.text.text_classifier import model_options as mo
from mediapipe.model_maker.python.text.text_classifier import model_spec as ms
from mediapipe.model_maker.python.text.text_classifier import model_with_tokenizer
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

  if isinstance(
      options.model_options, mo.AverageWordEmbeddingModelOptions
  ) and (
      options.supported_model
      != ms.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER
  ):
    raise ValueError(
        "Expected AVERAGE_WORD_EMBEDDING_CLASSIFIER,"
        f" got {options.supported_model}"
    )
  if isinstance(options.model_options, mo.BertModelOptions) and (
      not isinstance(options.supported_model.value(), ms.BertClassifierSpec)
  ):
    raise ValueError(
        f"Expected a Bert Classifier, got {options.supported_model}"
    )


class TextClassifier(classifier.Classifier):
  """API for creating and training a text classification model."""

  def __init__(
      self, model_spec: Any, label_names: Sequence[str], shuffle: bool
  ):
    super().__init__(
        model_spec=model_spec, label_names=label_names, shuffle=shuffle
    )
    self._model_spec = model_spec
    self._text_preprocessor: preprocessor.TextClassifierPreprocessor = None

  @classmethod
  def create(
      cls,
      train_data: text_ds.Dataset,
      validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions,
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
          f"validation data label names {validation_data.label_names}"
      )

    _validate(options)
    if options.model_options is None:
      options.model_options = options.supported_model.value().model_options

    if options.hparams is None:
      options.hparams = options.supported_model.value().hparams

    if isinstance(options.supported_model.value(), ms.BertClassifierSpec):
      text_classifier = _BertClassifier.create_bert_classifier(
          train_data, validation_data, options
      )
    elif isinstance(
        options.supported_model.value(), ms.AverageWordEmbeddingClassifierSpec
    ):
      text_classifier = _AverageWordEmbeddingClassifier.create_average_word_embedding_classifier(
          train_data, validation_data, options
      )
    else:
      raise ValueError(f"Unknown model {options.supported_model}")

    return text_classifier

  @classmethod
  def load_bert_classifier(
      cls,
      options: text_classifier_options.TextClassifierOptions,
      saved_model_path: str,
      label_names: Sequence[str],
  ) -> "TextClassifier":
    if not isinstance(options.supported_model.value(), ms.BertClassifierSpec):
      raise ValueError(
          "Only loading BertClassifier is supported, got:"
          f" {options.supported_model}"
      )
    return _BertClassifier.load_bert_classifier(
        options, saved_model_path, label_names
    )

  def evaluate(
      self, data: ds.Dataset, batch_size: int = 32, **kwargs: Any
  ) -> Any:
    """Overrides Classifier.evaluate().

    Args:
      data: Evaluation dataset. Must be a TextClassifier Dataset.
      batch_size: Number of samples per evaluation step.
      **kwargs: Additional keyword arguments to pass to `model.evaluate()` such
        as return_dict=True. More info can be found at
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate.

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

    with self._hparams.get_strategy().scope():
      return self._model.evaluate(dataset, **kwargs)

  def save_model(
      self,
      model_name: str = "saved_model",
  ):
    """Saves the model in SavedModel format.

    For more information, see https://www.tensorflow.org/guide/saved_model.

    Args:
      model_name: Name of the saved model.
    """
    tf.io.gfile.makedirs(self._hparams.export_dir)
    saved_model_file = os.path.join(self._hparams.export_dir, model_name)
    self._model.save(
        saved_model_file,
        include_optimizer=False,
        save_format="tf",
    )

  def export_model(
      self,
      model_name: str = "model.tflite",
      quantization_config: Optional[quantization.QuantizationConfig] = None,
  ):
    """Converts and saves the model to a TFLite file with metadata included.

    Note that only the TFLite file is needed for deployment. This function also
    saves a metadata.json file to the same directory as the TFLite file which
    can be used to interpret the metadata content in the TFLite file.

    Args:
      model_name: File name to save TFLite model with metadata. The full export
        path is {self._hparams.export_dir}/{model_name}.
      quantization_config: The configuration for model quantization.
    """
    tf.io.gfile.makedirs(self._hparams.export_dir)
    tflite_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, "metadata.json")

    self.save_model(model_name="saved_model")
    saved_model_file = os.path.join(self._hparams.export_dir, "saved_model")

    tflite_model = model_util.convert_to_tflite_from_file(
        saved_model_file, quantization_config=quantization_config
    )
    vocab_filepath = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    self._save_vocab(vocab_filepath)

    writer = self._get_metadata_writer(tflite_model, vocab_filepath)
    tflite_model_with_metadata, metadata_json = writer.populate()
    model_util.save_tflite(tflite_model_with_metadata, tflite_file)
    with tf.io.gfile.GFile(metadata_file, "w") as f:
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

  def __init__(
      self,
      model_spec: ms.AverageWordEmbeddingClassifierSpec,
      model_options: mo.AverageWordEmbeddingModelOptions,
      hparams: hp.AverageWordEmbeddingHParams,
      label_names: Sequence[str],
  ):
    super().__init__(model_spec, label_names, hparams.shuffle)
    self._model_options = model_options
    self._hparams = hparams
    self._callbacks = model_util.get_default_callbacks(self._hparams.export_dir)
    self._loss_function = "sparse_categorical_crossentropy"
    self._metric_functions = [
        "accuracy",
    ]
    self._text_preprocessor: (
        preprocessor.AverageWordEmbeddingClassifierPreprocessor
    ) = None

  @classmethod
  def create_average_word_embedding_classifier(
      cls,
      train_data: text_ds.Dataset,
      validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions,
  ) -> "_AverageWordEmbeddingClassifier":
    """Creates, trains, and returns an Average Word Embedding classifier.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Options for creating and training the text classifier.

    Returns:
      An Average Word Embedding classifier.
    """
    average_word_embedding_classifier = _AverageWordEmbeddingClassifier(
        model_spec=options.supported_model.value(),
        model_options=options.model_options,
        hparams=options.hparams,
        label_names=train_data.label_names,
    )
    average_word_embedding_classifier._create_and_train_model(
        train_data, validation_data
    )
    return average_word_embedding_classifier

  def _create_and_train_model(
      self, train_data: text_ds.Dataset, validation_data: text_ds.Dataset
  ):
    """Creates the Average Word Embedding classifier keras model and trains it.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    (processed_train_data, processed_validation_data) = (
        self._load_and_run_preprocessor(train_data, validation_data)
    )
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
            vocab_size=self._model_options.vocab_size,
        )
    )
    return self._text_preprocessor.preprocess(
        train_data
    ), self._text_preprocessor.preprocess(validation_data)

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
            delim_regex_pattern=self._DELIM_REGEX_PATTERN,
            vocab_file_path=vocab_filepath,
        ),
        labels=metadata_writer.Labels().add(list(self._label_names)),
    )


class _BertClassifier(TextClassifier):
  """APIs to help create and train a BERT-based text classifier."""

  _INITIALIZER_RANGE = 0.02

  def __init__(
      self,
      model_spec: ms.BertClassifierSpec,
      model_options: mo.BertModelOptions,
      hparams: hp.BertHParams,
      label_names: Sequence[str],
  ):
    super().__init__(model_spec, label_names, hparams.shuffle)
    self._hparams = hparams
    if self._hparams.monitor:
      monitor = f"val_{self._hparams.monitor}"
    elif self._num_classes == 2 or self._hparams.is_multilabel:
      monitor = "val_auc"  # auc is a binary or multilabel only metric
    else:
      monitor = "val_accuracy"
    self._callbacks = list(
        model_util.get_default_callbacks(
            self._hparams.export_dir, self._hparams.checkpoint_frequency
        )
    ) + [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(self._hparams.export_dir, "best_model"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=False,
        )
    ]
    self._model_options = model_options
    self._text_preprocessor: preprocessor.BertClassifierPreprocessor = None
    with self._hparams.get_strategy().scope():
      class_weights = (
          self._hparams.multiclass_loss_weights
          if self._num_classes > 2
          else None
      )
      if self._hparams.is_multilabel:
        self._loss_function = loss_functions.MaskedBinaryCrossentropy(
            class_weights=class_weights
        )
      else:
        self._loss_function = loss_functions.SparseFocalLoss(
            self._hparams.gamma, self._num_classes, class_weight=class_weights
        )
      self._metric_functions = self._create_metrics()

  @classmethod
  def create_bert_classifier(
      cls,
      train_data: text_ds.Dataset,
      validation_data: text_ds.Dataset,
      options: text_classifier_options.TextClassifierOptions,
  ) -> "_BertClassifier":
    """Creates, trains, and returns a BERT-based classifier.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Options for creating and training the text classifier.

    Returns:
      A BERT-based classifier.
    """
    bert_classifier = _BertClassifier(
        model_spec=options.supported_model.value(),
        model_options=options.model_options,
        hparams=options.hparams,
        label_names=train_data.label_names,
    )
    tf.io.gfile.makedirs(bert_classifier._hparams.export_dir)
    config_file = os.path.join(
        bert_classifier._hparams.export_dir, "config.txt"
    )
    with tf.io.gfile.GFile(config_file, "w") as f:
      f.write(str(options))
      f.write(f"\nlabel_names:{train_data.label_names}")
    bert_classifier._create_and_train_model(train_data, validation_data)
    return bert_classifier

  @classmethod
  def load_bert_classifier(
      cls,
      options: text_classifier_options.TextClassifierOptions,
      saved_model_path: str,
      label_names: Sequence[str],
  ) -> "_BertClassifier":
    bert_classifier = _BertClassifier(
        model_spec=options.supported_model.value(),
        model_options=options.model_options,
        hparams=options.hparams,
        label_names=label_names,
    )
    with bert_classifier._hparams.get_strategy().scope():
      bert_classifier._create_model()
      # create dummy optimizer so model compiles
      bert_classifier._optimizer = tfa_optimizers.LAMB(
          3e-4,
          weight_decay_rate=bert_classifier._hparams.weight_decay,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
          global_clipnorm=1.0,
      )
      bert_classifier._model = tf.keras.models.load_model(
          saved_model_path, compile=False
      )
    bert_classifier._load_preprocessor()
    bert_classifier._model.compile(
        optimizer=bert_classifier._optimizer,
        loss=bert_classifier._loss_function,
        weighted_metrics=bert_classifier._metric_functions,
    )
    return bert_classifier

  def _create_and_train_model(
      self, train_data: text_ds.Dataset, validation_data: text_ds.Dataset
  ):
    """Creates the BERT-based classifier keras model and trains it.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    self._load_preprocessor()
    (processed_train_data, processed_validation_data) = self._run_preprocessor(
        train_data, validation_data
    )
    with self._hparams.get_strategy().scope():
      self._create_model()
      self._create_optimizer(processed_train_data)
    self._train_model(
        processed_train_data,
        processed_validation_data,
        checkpoint_path=os.path.join(self._hparams.export_dir, "checkpoint"),
    )

  def _load_preprocessor(self):
    """Loads a BertClassifierPreprocessor."""
    self._text_preprocessor = preprocessor.BertClassifierPreprocessor(
        seq_len=self._model_options.seq_len,
        do_lower_case=self._model_spec.do_lower_case,
        uri=self._model_spec.get_path(),
        model_name=self._model_spec.name,
        tokenizer=self._hparams.tokenizer,
    )

  def _run_preprocessor(
      self, train_data: text_ds.Dataset, validation_data: text_ds.Dataset
  ) -> Tuple[text_ds.Dataset, text_ds.Dataset]:
    """Runs BertClassifierPreprocessor on the data.

    Args:
      train_data: Training data.
      validation_data: Validation data.

    Returns:
      Preprocessed training data and preprocessed validation data.
    """
    return (
        self._text_preprocessor.preprocess(train_data),
        self._text_preprocessor.preprocess(validation_data),
    )

  def _get_eligle_monitor_metric_variables(
      self,
  ) -> Tuple[Optional[str], Optional[Sequence[float]]]:
    """Returns the monitor metric name and class weights if eligible."""
    class_weights = self._hparams.best_checkpoint_monitor_weights
    monitor_name = self._hparams.monitor

    if (
        class_weights
        and monitor_name == "multiclass_recalls_accuracy_weighted_sum"
    ):
      return monitor_name, class_weights
    elif (
        not class_weights
        and monitor_name == "multiclass_recalls_accuracy_weighted_sum"
    ):
      raise ValueError(
          "best_checkpoint_monitor_weights must be specified for"
          " multiclass_recalls_accuracy_weighted_sum monitor metric."
      )
    elif (
        class_weights
        and monitor_name != "multiclass_recalls_accuracy_weighted_sum"
    ):
      raise ValueError(
          "best_checkpoint_monitor_weights can only be specified for"
          " multiclass_recalls_accuracy_weighted_sum monitor metric."
      )
    return None, None

  def _create_metrics(self):
    """Creates metrics for training and evaluation.

    The default metrics are accuracy, precision, and recall.

    For binary classification tasks only (num_classes=2):
      Users can configure PrecisionAtRecall and RecallAtPrecision metrics using
      the desired_precisions and desired_recalls fields in BertHParams.
      Users can also configure the desired_thresholds field to specify
      thresholds for the BinarySparsePrecision and BinarySparseRecall metrics.

    Returns:
      A list of tf.keras.Metric subclasses which can be used with model.compile
    """
    metric_functions = []
    if self._hparams.is_multilabel:
      metric_functions.append(tf.keras.metrics.BinaryAccuracy())
      metric_functions.append(
          tf.keras.metrics.AUC(
              name="auc", multi_label=True, num_thresholds=1000
          )
      )
      for i in range(self._num_classes):
        metric_functions.append(
            metrics.BinaryAUC(name=f"auc_{i}", num_thresholds=1000, class_id=i)
        )
        if self._hparams.desired_precisions:
          for desired_precision in self._hparams.desired_precisions:
            metric_functions.append(
                metrics.MaskedBinaryRecallAtPrecision(
                    desired_precision,
                    name=f"recall_at_precision_{desired_precision}_{i}",
                    num_thresholds=1000,
                    class_id=i,
                )
            )
        if self._hparams.desired_recalls:
          for desired_recall in self._hparams.desired_recalls:
            metric_functions.append(
                metrics.MaskedBinaryPrecisionAtRecall(
                    desired_recall,
                    name=f"precision_at_recall_{desired_recall}_{i}",
                    num_thresholds=1000,
                    class_id=i,
                )
            )
        if self._hparams.desired_thresholds:
          for desired_threshold in self._hparams.desired_thresholds:
            metric_functions.append(
                metrics.MaskedBinaryPrecision(
                    desired_threshold,
                    name=f"precision_at_{desired_threshold}_class{i}",
                    class_id=i,
                )
            )
            metric_functions.append(
                metrics.MaskedBinaryRecall(
                    desired_threshold,
                    name=f"recall_at_{desired_threshold}_class{i}",
                    class_id=i,
                )
            )
    else:
      metric_functions.append(
          tf.keras.metrics.SparseCategoricalAccuracy(
              "accuracy", dtype=tf.float32
          ),
      )
      if self._num_classes == 2:
        metric_functions.extend([
            metrics.BinarySparseAUC(name="auc", num_thresholds=1000),
        ])
        if self._hparams.desired_thresholds:
          for desired_threshold in self._hparams.desired_thresholds:
            metric_functions.append(
                metrics.BinarySparsePrecision(
                    name=f"precision_{desired_threshold}",
                    thresholds=desired_threshold,
                )
            )
            metric_functions.append(
                metrics.BinarySparseRecall(
                    name=f"recall_{desired_threshold}",
                    thresholds=desired_threshold,
                )
            )
        if self._hparams.desired_precisions:
          for desired_precision in self._hparams.desired_precisions:
            metric_functions.append(
                metrics.BinarySparseRecallAtPrecision(
                    desired_precision,
                    name=f"recall_at_precision_{desired_precision}",
                    num_thresholds=1000,
                )
            )
        if self._hparams.desired_recalls:
          for desired_recall in self._hparams.desired_recalls:
            metric_functions.append(
                metrics.BinarySparseRecallAtPrecision(
                    desired_recall,
                    name=f"precision_at_recall_{desired_recall}",
                    num_thresholds=1000,
                )
            )
      else:
        for i in range(self._num_classes):
          metric_functions.append(
              metrics.MultiClassSparsePrecision(
                  name=f"class_{i}_precision", class_id=i
              )
          )
          metric_functions.append(
              metrics.MultiClassSparseRecall(
                  name=f"class_{i}_recall", class_id=i
              )
          )
        monitor_name, class_weights = (
            self._get_eligle_monitor_metric_variables()
        )
        if monitor_name and class_weights:
          monitor_metrics = [
              tf.keras.metrics.SparseCategoricalAccuracy(
                  "accuracy", dtype=tf.float32
              )
          ]
          for i in range(self._num_classes):
            monitor_metrics.append(
                metrics.MultiClassSparseRecall(
                    name=f"class_{i}_recall", class_id=i
                ),
            )
          weights = [1.0 - sum(class_weights)]
          weights.extend(class_weights)
          metric_functions.append(
              metrics.WeightedSumMetric(
                  addend_metrics=monitor_metrics,
                  weights=weights,
                  name=monitor_name,
              )
          )
        if self._hparams.desired_precisions or self._hparams.desired_recalls:
          raise ValueError(
              "desired_recalls and desired_precisions parameters are binary"
              " metrics and not supported for num_classes > 2. Found"
              f" num_classes: {self._num_classes}"
          )
    return metric_functions

  def _create_model(self):
    """Creates a BERT-based classifier model.

    The model architecture consists of stacking a dense classification layer and
    dropout layer on top of the BERT encoder outputs.
    """
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(
            shape=(None,),
            dtype=tf.int32,
            name="input_word_ids",
        ),
        input_mask=tf.keras.layers.Input(
            shape=(None,),
            dtype=tf.int32,
            name="input_mask",
        ),
        input_type_ids=tf.keras.layers.Input(
            shape=(None,),
            dtype=tf.int32,
            name="input_type_ids",
        ),
    )
    if self._model_spec.is_tf2:
      encoder = hub.KerasLayer(
          self._model_spec.get_path(),
          trainable=self._model_options.do_fine_tuning,
          load_options=tf.saved_model.LoadOptions(
              experimental_io_device="/job:localhost"
          ),
      )
      encoder_outputs = encoder(encoder_inputs)
      pooled_output = encoder_outputs["pooled_output"]
    else:
      renamed_inputs = dict(
          input_ids=encoder_inputs["input_word_ids"],
          input_mask=encoder_inputs["input_mask"],
          segment_ids=encoder_inputs["input_type_ids"],
      )
      encoder = hub_loader.HubKerasLayerV1V2(
          self._model_spec.get_path(),
          signature="tokens",
          output_key="pooled_output",
          trainable=self._model_options.do_fine_tuning,
      )
      pooled_output = encoder(renamed_inputs)

    output = tf.keras.layers.Dropout(rate=self._model_options.dropout_rate)(
        pooled_output
    )
    initializer = tf.keras.initializers.TruncatedNormal(
        stddev=self._INITIALIZER_RANGE
    )
    output = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=initializer,
        name="output",
        activation="sigmoid" if self._hparams.is_multilabel else "softmax",
        dtype=tf.float32,
    )(output)
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
        train_data=train_data,
    )
    total_steps = self._hparams.steps_per_epoch * self._hparams.epochs
    warmup_steps = int(total_steps * 0.1)
    initial_lr = self._hparams.learning_rate
    # Implements linear decay of the learning rate.
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        end_learning_rate=self._hparams.end_learning_rate,
        power=1.0,
    )
    if warmup_steps:
      lr_schedule = model_util.WarmUp(
          initial_learning_rate=initial_lr,
          decay_schedule_fn=lr_schedule,
          warmup_steps=warmup_steps,
      )
    if self._hparams.optimizer == hp.BertOptimizer.ADAMW:
      self._optimizer = tf.keras.optimizers.experimental.AdamW(
          lr_schedule,
          weight_decay=self._hparams.weight_decay,
          epsilon=1e-6,
          global_clipnorm=1.0,
      )
      self._optimizer.exclude_from_weight_decay(
          var_names=["LayerNorm", "layer_norm", "bias"]
      )
    elif self._hparams.optimizer == hp.BertOptimizer.LAMB:
      self._optimizer = tfa_optimizers.LAMB(
          lr_schedule,
          weight_decay_rate=self._hparams.weight_decay,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
          global_clipnorm=1.0,
      )
    else:
      raise ValueError(
          "BertHParams.optimizer must be set to ADAM or "
          f"LAMB. Got {self._hparams.optimizer}."
      )

  def _save_vocab(self, vocab_filepath: str):
    tf.io.gfile.copy(
        self._text_preprocessor.get_vocab_file(), vocab_filepath, overwrite=True
    )

  def _get_metadata_writer(self, tflite_model: bytearray, vocab_filepath: str):
    return text_classifier_writer.MetadataWriter.create_for_bert_model(
        model_buffer=tflite_model,
        tokenizer=metadata_writer.BertTokenizer(vocab_filepath),
        labels=metadata_writer.Labels().add(list(self._label_names)),
        ids_name=self._model_spec.tflite_input_name["ids"],
        mask_name=self._model_spec.tflite_input_name["mask"],
        segment_name=self._model_spec.tflite_input_name["segment_ids"],
    )

  def export_model(
      self,
      model_name: str = "model.tflite",
      quantization_config: Optional[quantization.QuantizationConfig] = None,
  ):
    """Converts and saves the model to a TFLite file with metadata included.

    Note that only the TFLite file is needed for deployment. This function also
    saves a metadata.json file to the same directory as the TFLite file which
    can be used to interpret the metadata content in the TFLite file.

    This override method is needed to disable dynamic sequence length in the
    MediaPipe-wrapped model. See b/361090759 for more info.

    Args:
      model_name: File name to save TFLite model with metadata. The full export
        path is {self._hparams.export_dir}/{model_name}.
      quantization_config: The configuration for model quantization.
    """
    tf.io.gfile.makedirs(self._hparams.export_dir)
    tflite_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, "metadata.json")

    constant_len_inputs = dict(
        input_word_ids=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,),
            dtype=tf.int32,
            name="input_word_ids",
        ),
        input_mask=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,),
            dtype=tf.int32,
            name="input_mask",
        ),
        input_type_ids=tf.keras.layers.Input(
            shape=(self._model_options.seq_len,),
            dtype=tf.int32,
            name="input_type_ids",
        ),
    )
    output = self._model(constant_len_inputs)
    constant_len_model = tf.keras.Model(
        inputs=constant_len_inputs, outputs=output
    )
    saved_model_file = os.path.join(
        self._hparams.export_dir, "saved_model_constant_len"
    )
    constant_len_model.save(
        saved_model_file,
        include_optimizer=False,
        save_format="tf",
    )

    tflite_model = model_util.convert_to_tflite_from_file(
        saved_model_file, quantization_config=quantization_config
    )
    vocab_filepath = os.path.join(tempfile.mkdtemp(), "vocab.txt")
    self._save_vocab(vocab_filepath)

    writer = self._get_metadata_writer(tflite_model, vocab_filepath)
    tflite_model_with_metadata, metadata_json = writer.populate()
    model_util.save_tflite(tflite_model_with_metadata, tflite_file)
    with tf.io.gfile.GFile(metadata_file, "w") as f:
      f.write(metadata_json)

  def export_model_with_tokenizer(
      self,
      model_name: str = "model_with_tokenizer.tflite",
      quantization_config: Optional[quantization.QuantizationConfig] = None,
  ):
    """Converts and saves the model to a TFLite file with the tokenizer.

    Note that unlike the export_model method, this export method will include
    a FastBertTokenizer in the TFLite graph. The resulting TFLite will not have
    metadata information to use with MediaPipe Tasks, but can be run directly
    using TFLite Inference: https://www.tensorflow.org/lite/guide/inference

    For more information on the tokenizer, see:
      https://www.tensorflow.org/text/api_docs/python/text/FastBertTokenizer

    Args:
      model_name: File name to save TFLite model with tokenizer. The full export
        path is {self._hparams.export_dir}/{model_name}.
      quantization_config: The configuration for model quantization.
    """
    tf.io.gfile.makedirs(self._hparams.export_dir)
    tflite_file = os.path.join(self._hparams.export_dir, model_name)
    if (
        self._hparams.tokenizer
        != bert_tokenizer.SupportedBertTokenizers.FAST_BERT_TOKENIZER
    ):
      print(
          f"WARNING: This model was trained with {self._hparams.tokenizer} "
          "tokenizer, but the exported model with tokenizer will have a "
          f"{bert_tokenizer.SupportedBertTokenizers.FAST_BERT_TOKENIZER} "
          "tokenizer."
      )
      tokenizer = bert_tokenizer.BertFastTokenizer(
          vocab_file=self._text_preprocessor.get_vocab_file(),
          do_lower_case=self._model_spec.do_lower_case,
          seq_len=self._model_options.seq_len,
      )
    else:
      tokenizer = self._text_preprocessor.tokenizer

    model = model_with_tokenizer.ModelWithTokenizer(tokenizer, self._model)
    model(tf.constant(["Example input data".encode("utf-8")]))  # build model
    saved_model_file = os.path.join(
        self._hparams.export_dir, "saved_model_with_tokenizer"
    )
    model.save(saved_model_file)
    tflite_model = model_util.convert_to_tflite_from_file(
        saved_model_file,
        quantization_config=quantization_config,
        allow_custom_ops=True,
    )
    model_util.save_tflite(tflite_model, tflite_file)
