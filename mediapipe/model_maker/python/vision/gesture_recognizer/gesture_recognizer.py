# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
      validation_data: Validation data. If None, skips validation process.
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

    gesture_recognizer._create_model()

    train_dataset = train_data.gen_tf_dataset(
        batch_size=options.hparams.batch_size,
        is_training=True,
        shuffle=options.hparams.shuffle)
    options.hparams.steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=options.hparams.steps_per_epoch,
        batch_size=options.hparams.batch_size,
        train_data=train_data)
    train_dataset = train_dataset.take(count=options.hparams.steps_per_epoch)

    validation_dataset = validation_data.gen_tf_dataset(
        batch_size=options.hparams.batch_size, is_training=False)

    tf.compat.v1.logging.info('Training the gesture recognizer model...')
    gesture_recognizer._train(
        train_data=train_dataset, validation_data=validation_dataset)

    return gesture_recognizer

  def _train(self, train_data: tf.data.Dataset,
             validation_data: tf.data.Dataset):
    """Trains the model with input train_data.

    The training results are recorded by a self.History object returned by
    tf.keras.Model.fit().

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    hparams = self._hparams

    scheduler = lambda epoch: hparams.learning_rate * (hparams.lr_decay**epoch)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    job_dir = hparams.export_dir
    checkpoint_path = os.path.join(job_dir, 'epoch_models')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_path, 'model-{epoch:04d}'),
        save_weights_only=True)

    best_model_path = os.path.join(job_dir, 'best_model_weights')
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'))

    self._model.compile(
        optimizer='adam',
        loss=loss_functions.FocalLoss(gamma=self._hparams.gamma),
        metrics=['categorical_accuracy'])

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint:
      print(f'Resuming from {latest_checkpoint}')
      self._model.load_weights(latest_checkpoint)

    self._history = self._model.fit(
        x=train_data,
        epochs=hparams.epochs,
        validation_data=validation_data,
        validation_freq=1,
        callbacks=[
            checkpoint_callback, best_model_callback, scheduler_callback,
            tensorboard_callback
        ],
    )

  def _create_model(self):
    """Creates the hand gesture recognizer model.

    The gesture embedding model is pretrained and loaded from a tf.saved_model.
    """
    inputs = tf.keras.Input(
        shape=[self.embedding_size],
        batch_size=None,
        dtype=tf.float32,
        name='hand_embedding')

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.ReLU()(x)
    dropout_rate = self._model_options.dropout_rate
    x = tf.keras.layers.Dropout(rate=dropout_rate, name='dropout')(x)
    outputs = tf.keras.layers.Dense(
        self._num_classes,
        activation='softmax',
        name='custom_gesture_recognizer')(
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
        constants.GESTURE_EMBEDDER_TFLITE_MODEL_FILE)
    hand_detector_model_buffer = model_util.load_tflite_model_buffer(
        constants.HAND_DETECTOR_TFLITE_MODEL_FILE)
    hand_landmarks_detector_model_buffer = model_util.load_tflite_model_buffer(
        constants.HAND_LANDMARKS_DETECTOR_TFLITE_MODEL_FILE)
    canned_gesture_model_buffer = model_util.load_tflite_model_buffer(
        constants.CANNED_GESTURE_CLASSIFIER_TFLITE_MODEL_FILE)

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
