# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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
"""APIs to train face stylization model."""

import os
from typing import Callable, Optional

import numpy as np
import tensorflow as tf

from mediapipe.model_maker.python.core.data import classification_dataset as classification_ds
from mediapipe.model_maker.python.core.utils import loss_functions
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.vision.core import image_preprocessing
from mediapipe.model_maker.python.vision.face_stylizer import constants
from mediapipe.model_maker.python.vision.face_stylizer import face_stylizer_options
from mediapipe.model_maker.python.vision.face_stylizer import hyperparameters as hp
from mediapipe.model_maker.python.vision.face_stylizer import model_options as model_opt
from mediapipe.model_maker.python.vision.face_stylizer import model_spec as ms


class FaceStylizer(object):
  """FaceStylizer for building face stylization model.

  Attributes:
    w_avg: An average face latent code to regularize face generation in face
      stylization.
  """

  def __init__(
      self,
      model_spec: ms.ModelSpec,
      model_options: model_opt.FaceStylizerModelOptions,
      hparams: hp.HParams,
  ):
    """Initializes face stylizer.

    Args:
      model_spec: Specification for the model.
      model_options: Model options for creating face stylizer.
      hparams: The hyperparameters for training face stylizer.
    """
    self._model_spec = model_spec
    self._model_options = model_options
    self._hparams = hparams
    # TODO: Support face alignment in image preprocessor.
    self._preprocessor = image_preprocessing.Preprocessor(
        input_shape=self._model_spec.input_image_shape,
        num_classes=1,
        mean_rgb=self._model_spec.mean_rgb,
        stddev_rgb=self._model_spec.stddev_rgb,
    )

  @classmethod
  def create(
      cls,
      train_data: classification_ds.ClassificationDataset,
      options: face_stylizer_options.FaceStylizerOptions,
  ) -> 'FaceStylizer':
    """Creates and trains a face stylizer with input datasets.

    Args:
      train_data: The input style image dataset for training the face stylizer.
      options: The options to configure face stylizer.

    Returns:
      A FaceStylizer instant with the trained model.
    """
    if options.model_options is None:
      options.model_options = model_opt.FaceStylizerModelOptions()

    if options.hparams is None:
      options.hparams = hp.HParams()

    spec = ms.SupportedModels.get(options.model)

    face_stylizer = cls(
        model_spec=spec,
        model_options=options.model_options,
        hparams=options.hparams,
    )
    face_stylizer._create_and_train_model(train_data)
    return face_stylizer

  def _create_and_train_model(
      self, train_data: classification_ds.ClassificationDataset
  ):
    """Creates and trains the face stylizer model.

    Args:
      train_data: Training data.
    """
    self._create_model()
    self._train_model(train_data=train_data, preprocessor=self._preprocessor)

  def _create_model(self):
    """Creates the componenets of face stylizer."""
    self._encoder = model_util.load_keras_model(
        constants.FACE_STYLIZER_ENCODER_MODEL_FILES.get_path()
    )
    self._decoder = model_util.load_keras_model(
        constants.FACE_STYLIZER_DECODER_MODEL_FILES.get_path()
    )
    self._mapping_network = model_util.load_keras_model(
        constants.FACE_STYLIZER_MAPPING_MODEL_FILES.get_path()
    )
    self._discriminator = model_util.load_keras_model(
        constants.FACE_STYLIZER_DISCRIMINATOR_MODEL_FILES.get_path()
    )
    with tf.io.gfile.GFile(
        constants.FACE_STYLIZER_W_FILES.get_path(), 'rb'
    ) as f:
      w_avg = np.load(f)

    self.w_avg = w_avg[: self._model_spec.style_block_num][np.newaxis]

  def _train_model(
      self,
      train_data: classification_ds.ClassificationDataset,
      preprocessor: Optional[Callable[..., bool]] = None,
  ):
    """Trains the face stylizer model.

    Args:
      train_data: The data for training model.
      preprocessor: The image preprocessor.
    """
    train_dataset = train_data.gen_tf_dataset(preprocess=preprocessor)

    # TODO: Support processing mulitple input style images. The
    # input style images are expected to have similar style.
    # style_sample represents a tuple of (style_image, style_label).
    style_sample = next(iter(train_dataset))
    style_img = style_sample[0]

    batch_size = self._hparams.batch_size
    label_in = tf.zeros(shape=[batch_size, 0])

    style_encoding = self._encoder(style_img)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self._hparams.learning_rate,
        beta_1=self._hparams.beta_1,
        beta_2=self._hparams.beta_2,
    )

    image_perceptual_quality_loss = loss_functions.ImagePerceptualQualityLoss(
        loss_weight=self._model_options.perception_loss_weight
    )

    for i in range(self._hparams.epochs):
      noise = tf.random.normal(shape=[batch_size, constants.STYLE_DIM])

      mean_w = self._mapping_network([noise, label_in], training=False)[
          :, : self._model_spec.style_block_num
      ]
      style_encodings = tf.tile(style_encoding, [batch_size, 1, 1])

      in_latent = tf.Variable(tf.identity(style_encodings))

      alpha = self._model_options.alpha
      for swap_layer in self._model_options.swap_layers:
        in_latent = in_latent[:, swap_layer].assign(
            alpha * style_encodings[:, swap_layer]
            + (1 - alpha) * mean_w[:, swap_layer]
        )

      with tf.GradientTape() as tape:
        outputs = self._decoder(
            {'inputs': in_latent + self.w_avg},
            training=True,
        )
        gen_img = outputs['image'][-1]

        real_feature = self._discriminator(
            [tf.transpose(style_img, [0, 3, 1, 2]), label_in]
        )
        gen_feature = self._discriminator(
            [tf.transpose(gen_img, [0, 3, 1, 2]), label_in]
        )

        style_loss = image_perceptual_quality_loss(gen_img, style_img)
        style_loss += (
            tf.keras.losses.MeanAbsoluteError()(real_feature, gen_feature)
            * self._model_options.adv_loss_weight
        )
        tf.compat.v1.logging.info(f'Iteration {i} loss: {style_loss.numpy()}')

        tvars = self._decoder.trainable_variables
        grads = tape.gradient(style_loss, tvars)
        optimizer.apply_gradients(list(zip(grads, tvars)))

  # TODO: Add a metadata writer for face sytlizer model.
  def export_model(self, model_name: str = 'model.tflite'):
    """Converts and saves the model to a TFLite file with metadata included.

    Note that only the TFLite file is needed for deployment. This function
    also saves a metadata.json file to the same directory as the TFLite file
    which can be used to interpret the metadata content in the TFLite file.

    Args:
      model_name: File name to save TFLite model with metadata. The full export
        path is {self._hparams.export_dir}/{model_name}.
    """
    if not tf.io.gfile.exists(self._hparams.export_dir):
      tf.io.gfile.makedirs(self._hparams.export_dir)
    tflite_file = os.path.join(self._hparams.export_dir, model_name)

    # Create an end-to-end model by concatenating encoder and decoder
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = self._encoder(inputs)
    x = self._decoder({'inputs': x + self.w_avg})
    x = x['image'][-1]
    # Scale the data range from [-1, 1] to [0, 1] to support running inference
    # on both CPU and GPU.
    outputs = (x + 1.0) / 2.0
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    tflite_model = model_util.convert_to_tflite(
        model=model,
        preprocess=self._preprocessor,
    )
    model_util.save_tflite(tflite_model, tflite_file)
