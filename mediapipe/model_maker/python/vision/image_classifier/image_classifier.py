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
        use_augmentation=hparams.do_data_augmentation,
        one_hot=hparams.one_hot,
    )
    self._callbacks = model_util.get_default_callbacks(
        self._hparams.export_dir, self._hparams.checkpoint_frequency
    )

    if not self._hparams.multi_labels:
      self._loss_function = tf.keras.losses.CategoricalCrossentropy(
          label_smoothing=self._hparams.label_smoothing
      )
      self._metric_functions = ['accuracy']
    else:
      self._loss_function = tf.keras.losses.BinaryCrossentropy()
      self._metric_functions = [
          tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.Recall(thresholds=0.25, name='Recall_0.25'),
          tf.keras.metrics.Recall(thresholds=0.5, name='Recall_0.5'),
          tf.keras.metrics.Recall(thresholds=0.75, name='Recall_0.75'),
          tf.keras.metrics.Precision(thresholds=0.25, name='Precision_0.25'),
          tf.keras.metrics.Precision(thresholds=0.5, name='Precision_0.5'),
          tf.keras.metrics.Precision(thresholds=0.75, name='Precision_0.75'),
          tf.keras.metrics.AUC(),
      ]
      if self._num_classes > 1:
        for i in range(self._num_classes):
          self._metric_functions.extend([
              tf.keras.metrics.Recall(
                  thresholds=0.25, name=f'Recall_0.25_{i}', class_id=i
              ),
              tf.keras.metrics.Recall(
                  thresholds=0.5, name=f'Recall_0.5_{i}', class_id=i
              ),
              tf.keras.metrics.Recall(
                  thresholds=0.75, name=f'Recall_0.75_{i}', class_id=i
              ),
              tf.keras.metrics.Precision(
                  thresholds=0.25, name=f'Precision_0.25_{i}', class_id=i
              ),
              tf.keras.metrics.Precision(
                  thresholds=0.5, name=f'Precision_0.5_{i}', class_id=i
              ),
              tf.keras.metrics.Precision(
                  thresholds=0.75, name=f'Precision_0.75_{i}', class_id=i
              ),
          ])
    self._history = None  # Training history returned from `keras_model.fit`.

  @classmethod
  def create(
      cls,
      train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset,
      options: image_classifier_options.ImageClassifierOptions,
  ) -> 'ImageClassifier':
    """Creates and trains an ImageClassifier.

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

    with options.hparams.get_strategy().scope():
      spec = ms.SupportedModels.get(options.supported_model)
      image_classifier = cls(
          model_spec=spec,
          label_names=train_data.label_names,
          hparams=options.hparams,
          model_options=options.model_options,
      )

      image_classifier._create_and_train_model(train_data, validation_data)
    return image_classifier

  def _create_and_train_model(
      self, train_data: classification_ds.ClassificationDataset,
      validation_data: classification_ds.ClassificationDataset):
    """Creates and trains the model and optimizer.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    self._create_model()

    ckpt_path = tf.train.latest_checkpoint(
        self._hparams.export_dir + '/checkpoint'
    )
    if ckpt_path is not None:
      self._model.load_weights(ckpt_path)
    self._hparams.steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=self._hparams.steps_per_epoch,
        batch_size=self._hparams.batch_size,
        train_data=train_data,
    )
    self._optimizer = self._create_optimizer()
    self._train_model(
        train_data=train_data,
        validation_data=validation_data,
        preprocessor=self._preprocess,
        checkpoint_path=os.path.join(self._hparams.export_dir, 'checkpoint'),
    )

  def _create_model(self):
    """Creates the classifier model from TFHub pretrained models."""
    image_size = self._model_spec.input_image_shape

    if self._model_spec.name == 'mobilenet_v2_keras':
      inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
      mobilenet_v2_layer = tf.keras.applications.MobileNetV2(
          alpha=self._model_options.alpha,
          weights='imagenet',
          include_top=False,
          pooling=None,
      )
      outputs = mobilenet_v2_layer(inputs)
      outputs = tf.keras.layers.AveragePooling2D(
          pool_size=(7, 7),
          strides=(1, 1),
          padding='valid',
          name='AvgPool_7x7',
      )(outputs)
      num_fcs = 2
      num_units = 512
      for fc_id in range(num_fcs):
        outputs = tf.keras.layers.Conv2D(
            num_units,
            1,
            padding='same',
            activation='relu6',
            name=f'FC_{fc_id}',
        )(outputs)
      outputs = tf.keras.layers.Flatten()(outputs)
      outputs = tf.keras.layers.Dropout(rate=self._model_options.dropout_rate)(
          outputs
      )
      outputs = tf.keras.layers.Dense(
          self._num_classes,
          activation='sigmoid',
          name='logits',
          kernel_regularizer=tf.keras.regularizers.l1_l2(
              l1=self._hparams.l1_regularizer,
              l2=self._hparams.l2_regularizer,
          ),
      )(outputs)
      self._model = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
      module_layer = hub.KerasLayer(
          handle=self._model_spec.uri, trainable=self._hparams.do_fine_tuning
      )
      self._model = tf.keras.Sequential([
          tf.keras.Input(shape=(image_size[0], image_size[1], 3)),
          module_layer,
          tf.keras.layers.Dropout(rate=self._model_options.dropout_rate),
          tf.keras.layers.Dense(
              units=self._num_classes,
              activation='softmax',
              kernel_regularizer=tf.keras.regularizers.l1_l2(
                  l1=self._hparams.l1_regularizer,
                  l2=self._hparams.l2_regularizer,
              ),
          ),
      ])
    print(self._model.summary())

  def export_model(
      self,
      model_name: str = 'model.tflite',
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

  def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
    """Creates an optimizer with learning rate schedule.

    Uses Keras CosineDecay schedule for the learning rate by default.

    Returns:
      A tf.keras.optimizers.Optimizer for model training.
    """
    # Learning rate is linear to batch size.
    init_lr = self._hparams.learning_rate * self._hparams.batch_size / 256

    # Get decay steps.
    total_training_steps = self._hparams.steps_per_epoch * self._hparams.epochs
    default_decay_steps = (
        self._hparams.decay_samples // self._hparams.batch_size)
    decay_steps = max(total_training_steps, default_decay_steps)

    learning_rate_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=init_lr, decay_steps=decay_steps, alpha=0.0)
    warmup_steps = self._hparams.warmup_epochs * self._hparams.steps_per_epoch
    if warmup_steps:
      learning_rate_fn = model_util.WarmUp(
          initial_learning_rate=init_lr,
          decay_schedule_fn=learning_rate_fn,
          warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate_fn, rho=0.9, momentum=0.9, epsilon=0.001)

    return optimizer
