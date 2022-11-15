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
import os

from typing import List, Optional

import tensorflow as tf
import tensorflow_hub as hub

from mediapipe.model_maker.python.core.data import classification_dataset as classification_ds
from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.vision.core import image_preprocessing
from mediapipe.model_maker.python.vision.image_classifier import hyperparameters as hp
from mediapipe.model_maker.python.vision.image_classifier import image_classifier_options
from mediapipe.model_maker.python.vision.image_classifier import model_options as model_opt
from mediapipe.model_maker.python.vision.image_classifier import model_spec as ms
from mediapipe.model_maker.python.vision.image_classifier import train_image_classifier_lib
from mediapipe.tasks.python.metadata.metadata_writers import image_classifier as image_classifier_writer
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer


class ImageClassifier(classifier.Classifier):
  """ImageClassifier for building image classification model."""

  def __init__(self, model_spec: ms.ModelSpec, label_names: List[str],
               hparams: hp.HParams,
               model_options: model_opt.ImageClassifierModelOptions):
    """Initializes ImageClassifier class.

    Args:
      model_spec: Specification for the model.
      label_names: A list of label names for the classes.
      hparams: The hyperparameters for training image classifier.
      model_options: Model options for creating image classifier.
    """
    super().__init__(
        model_spec=model_spec, label_names=label_names, shuffle=hparams.shuffle)
    self._hparams = hparams
    self._model_options = model_options
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
      train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset,
      options: image_classifier_options.ImageClassifierOptions,
  ) -> 'ImageClassifier':
    """Creates and trains an image classifier.

    Loads data and trains the model based on data for image classification. If a
    checkpoint file exists in the {options.hparams.export_dir}/checkpoint/
    directory, the training process will load the weight from the checkpoint
    file for continual training.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: configuration to create image classifier.

    Returns:
      An instance based on ImageClassifier.
    """
    if options.hparams is None:
      options.hparams = hp.HParams()

    if options.model_options is None:
      options.model_options = model_opt.ImageClassifierModelOptions()

    spec = ms.SupportedModels.get(options.supported_model)
    image_classifier = cls(
        model_spec=spec,
        label_names=train_data.label_names,
        hparams=options.hparams,
        model_options=options.model_options)

    image_classifier._create_model()

    tf.compat.v1.logging.info('Training the models...')
    image_classifier._train(
        train_data=train_data, validation_data=validation_data)

    return image_classifier

  # TODO: Migrate to the shared training library of Model Maker.
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
        tf.keras.layers.Dropout(rate=self._model_options.dropout_rate),
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
    metadata_file = os.path.join(self._hparams.export_dir, 'metadata.json')

    tflite_model = model_util.convert_to_tflite(
        model=self._model,
        quantization_config=quantization_config,
        preprocess=self._preprocess)
    writer = image_classifier_writer.MetadataWriter.create(
        tflite_model,
        self._model_spec.mean_rgb,
        self._model_spec.stddev_rgb,
        labels=metadata_writer.Labels().add(list(self._label_names)))
    tflite_model_with_metadata, metadata_json = writer.populate()
    model_util.save_tflite(tflite_model_with_metadata, tflite_file)
    with open(metadata_file, 'w') as f:
      f.write(metadata_json)
