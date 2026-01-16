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
"""APIs to train gesture recognizer model."""

import os
from typing import List

import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset as classification_ds
from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import loss_functions
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.vision.gesture_recognizer import constants
from mediapipe.model_maker.python.vision.gesture_recognizer import gesture_recognizer_options
from mediapipe.model_maker.python.vision.gesture_recognizer import hyperparameters as hp
from mediapipe.model_maker.python.vision.gesture_recognizer import metadata_writer
from mediapipe.model_maker.python.vision.gesture_recognizer import model_options as model_opt
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer as base_metadata_writer

_EMBEDDING_SIZE = 128


class GestureRecognizer(classifier.Classifier):
  """GestureRecognizer for building hand gesture recognizer model.

  Attributes:
    embedding_size: Size of the input gesture embedding vector.
  """

  def __init__(self, label_names: List[str],
               model_options: model_opt.GestureRecognizerModelOptions,
               hparams: hp.HParams):
    """Initializes GestureRecognizer class.

    Args:
      label_names: A list of label names for the classes.
      model_options: options to create gesture recognizer model.
      hparams: The hyperparameters for training hand gesture recognizer model.
    """
    super().__init__(
        model_spec=None, label_names=label_names, shuffle=hparams.shuffle)
    self._model_options = model_options
    self._hparams = hparams
    self._loss_function = loss_functions.FocalLoss(gamma=self._hparams.gamma)
    self._metric_functions = ['categorical_accuracy']
    self._optimizer = 'adam'
    self._callbacks = self._get_callbacks()
    self._history = None
    self.embedding_size = _EMBEDDING_SIZE

  @classmethod
  def create(
      cls,
      train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset,
      options: gesture_recognizer_options.GestureRecognizerOptions,
  ) -> 'GestureRecognizer':
    """Creates and trains a hand gesture recognizer with input datasets.

    If a checkpoint file exists in the {options.hparams.export_dir}/checkpoint/
    directory, the training process will load the weight from the checkpoint
    file for continual training.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: options for creating and training gesture recognizer model.

    Returns:
      An instance of GestureRecognizer.
    """
    if options.model_options is None:
      options.model_options = model_opt.GestureRecognizerModelOptions()

    if options.hparams is None:
      options.hparams = hp.HParams()

    gesture_recognizer = cls(
        label_names=train_data.label_names,
        model_options=options.model_options,
        hparams=options.hparams)
    gesture_recognizer._create_and_train_model(train_data, validation_data)
    return gesture_recognizer

  def _create_and_train_model(
      self,
      train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset,
  ):
    """Creates and trains the model.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    self._create_model()
    self._train_model(
        train_data=train_data,
        validation_data=validation_data,
        checkpoint_path=self._get_checkpoint_path(),
    )

  def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
    """Gets the list of callbacks to use in model training."""
    hparams = self._hparams

    scheduler = lambda epoch: hparams.learning_rate * (hparams.lr_decay**epoch)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    job_dir = hparams.export_dir
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(self._get_checkpoint_path(), 'model-{epoch:04d}'),
        save_weights_only=True,
    )

    best_model_path = os.path.join(job_dir, 'best_model_weights')
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'))
    return [
        checkpoint_callback,
        best_model_callback,
        scheduler_callback,
        tensorboard_callback,
    ]

  def _get_checkpoint_path(self) -> str:
    return os.path.join(self._hparams.export_dir, 'epoch_models')

  def _create_model(self):
    """Creates the hand gesture recognizer model.

    The gesture embedding model is pretrained and loaded from a tf.saved_model.
    """
    inputs = tf.keras.Input(
        shape=[self.embedding_size],
        batch_size=None,
        dtype=tf.float32,
        name='hand_embedding',
    )
    x = inputs
    dropout_rate = self._model_options.dropout_rate
    for i, width in enumerate(self._model_options.layer_widths):
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.ReLU()(x)
      x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
      x = tf.keras.layers.Dense(width, name=f'custom_gesture_recognizer_{i}')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    outputs = tf.keras.layers.Dense(
        self._num_classes,
        activation='softmax',
        name='custom_gesture_recognizer_out')(
            x)

    self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(self._model.summary())

  def export_model(self, model_name: str = 'gesture_recognizer.task'):
    """Converts the model to TFLite and exports as a model bundle file.

    Saves a model bundle file and metadata json file to hparams.export_dir. The
    resulting model bundle file will contain necessary models for hand
    detection, canned gesture classification, and customized gesture
    classification. Only the model bundle file is needed for the downstream
    gesture recognition task. The metadata.json file is saved only to
    interpret the contents of the model bundle file.

    The customized gesture model is in float without quantization. The model is
    lightweight and there is no need to balance performance and efficiency by
    quantization. The default score_thresholding is set to 0.5 as it can be
    adjusted during inference.

    Args:
      model_name: File name to save model bundle file. The full export path is
        {export_dir}/{model_name}.
    """
    # TODO: Convert keras embedder model instead of using tflite
    gesture_embedding_model_buffer = model_util.load_tflite_model_buffer(
        constants.GESTURE_EMBEDDER_TFLITE_MODEL_FILE.get_path()
    )
    hand_detector_model_buffer = model_util.load_tflite_model_buffer(
        constants.HAND_DETECTOR_TFLITE_MODEL_FILE.get_path()
    )
    hand_landmarks_detector_model_buffer = model_util.load_tflite_model_buffer(
        constants.HAND_LANDMARKS_DETECTOR_TFLITE_MODEL_FILE.get_path()
    )
    canned_gesture_model_buffer = model_util.load_tflite_model_buffer(
        constants.CANNED_GESTURE_CLASSIFIER_TFLITE_MODEL_FILE.get_path()
    )

    if not tf.io.gfile.exists(self._hparams.export_dir):
      tf.io.gfile.makedirs(self._hparams.export_dir)
    model_bundle_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, 'metadata.json')

    gesture_classifier_options = metadata_writer.GestureClassifierOptions(
        model_buffer=model_util.convert_to_tflite(self._model),
        labels=base_metadata_writer.Labels().add(list(self._label_names)),
        score_thresholding=base_metadata_writer.ScoreThresholding(
            global_score_threshold=0.5))

    writer = metadata_writer.MetadataWriter.create(
        hand_detector_model_buffer, hand_landmarks_detector_model_buffer,
        gesture_embedding_model_buffer, canned_gesture_model_buffer,
        gesture_classifier_options)
    model_bundle_content, metadata_json = writer.populate()
    with open(model_bundle_file, 'wb') as f:
      f.write(model_bundle_content)
    with open(metadata_file, 'w') as f:
      f.write(metadata_json)
