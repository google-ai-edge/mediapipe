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
"""APIs to train image classifier model."""

from typing import Any, List, Optional

import tensorflow as tf
import tensorflow_hub as hub

from mediapipe.model_maker.python.core.data import classification_dataset as classification_ds
from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import image_preprocessing
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.vision.image_classifier import hyperparameters as hp
from mediapipe.model_maker.python.vision.image_classifier import model_spec as ms
from mediapipe.model_maker.python.vision.image_classifier import train_image_classifier_lib


class ImageClassifier(classifier.Classifier):
  """ImageClassifier for building image classification model."""

  def __init__(self, model_spec: ms.ModelSpec, index_to_label: List[Any],
               hparams: hp.HParams):
    """Initializes ImageClassifier class.

    Args:
      model_spec: Specification for the model.
      index_to_label: A list that maps from index to label class name.
      hparams: The hyperparameters for training image classifier.
    """
    super(ImageClassifier, self).__init__(
        model_spec=model_spec,
        index_to_label=index_to_label,
        shuffle=hparams.shuffle,
        full_train=hparams.do_fine_tuning)
    self._hparams = hparams
    self._preprocess = image_preprocessing.Preprocessor(
        input_shape=self._model_spec.input_image_shape,
        num_classes=self._num_classes,
        mean_rgb=self._model_spec.mean_rgb,
        stddev_rgb=self._model_spec.stddev_rgb,
        use_augmentation=hparams.do_data_augmentation)
    self._history = None  # Training history returned from `keras_model.fit`.

  @classmethod
  def create(
      cls,
      model_spec: ms.SupportedModels,
      train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset,
      hparams: Optional[hp.HParams] = None,
  ) -> 'ImageClassifier':
    """Creates and trains an image classifier.

    Loads data and trains the model based on data for image classification.

    Args:
      model_spec: Specification for the model.
      train_data: Training data.
      validation_data: Validation data.
      hparams: Hyperparameters for training image classifier.

    Returns:
      An instance based on ImageClassifier.
    """
    if hparams is None:
      hparams = hp.HParams()

    spec = ms.SupportedModels.get(model_spec)
    image_classifier = cls(
        model_spec=spec,
        index_to_label=train_data.index_to_label,
        hparams=hparams)

    image_classifier._create_model()

    tf.compat.v1.logging.info('Training the models...')
    image_classifier._train(
        train_data=train_data, validation_data=validation_data)

    return image_classifier

  def _train(self, train_data: classification_ds.ClassificationDataset,
             validation_data: classification_ds.ClassificationDataset):
    """Trains the model with input train_data.

    The training results are recorded by a self._history object returned by
    tf.keras.Model.fit().

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """

    tf.compat.v1.logging.info('Training the models...')
    hparams = self._hparams
    if len(train_data) < hparams.batch_size:
      raise ValueError('The size of the train_data (%d) couldn\'t be smaller '
                       'than batch_size (%d). To solve this problem, set '
                       'the batch_size smaller or increase the size of the '
                       'train_data.' % (len(train_data), hparams.batch_size))

    train_dataset = train_data.gen_tf_dataset(
        batch_size=hparams.batch_size,
        is_training=True,
        shuffle=self._shuffle,
        preprocess=self._preprocess)
    hparams.steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=hparams.steps_per_epoch,
        batch_size=hparams.batch_size,
        train_data=train_data)
    train_dataset = train_dataset.take(count=hparams.steps_per_epoch)

    validation_dataset = validation_data.gen_tf_dataset(
        batch_size=hparams.batch_size,
        is_training=False,
        preprocess=self._preprocess)

    # Train the model.
    self._history = train_image_classifier_lib.train_model(
        model=self._model,
        hparams=hparams,
        train_ds=train_dataset,
        validation_ds=validation_dataset)

  def _create_model(self):
    """Creates the classifier model from TFHub pretrained models."""
    module_layer = hub.KerasLayer(
        handle=self._model_spec.uri, trainable=self._hparams.do_fine_tuning)

    image_size = self._model_spec.input_image_shape

    self._model = tf.keras.Sequential([
        tf.keras.Input(shape=(image_size[0], image_size[1], 3)), module_layer,
        tf.keras.layers.Dropout(rate=self._hparams.dropout_rate),
        tf.keras.layers.Dense(
            units=self._num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l1_l2(
                l1=self._hparams.l1_regularizer,
                l2=self._hparams.l2_regularizer))
    ])
    print(self._model.summary())

  def export_model(
      self,
      model_name: str = 'model.tflite',
      quantization_config: Optional[quantization.QuantizationConfig] = None):
    """Converts the model to the requested formats and exports to a file.

    Args:
      model_name: File name to save tflite model. The full export path is
        {export_dir}/{tflite_filename}.
      quantization_config: The configuration for model quantization.
    """
    super().export_tflite(
        self._hparams.model_dir,
        model_name,
        quantization_config,
        preprocess=self._preprocess)
