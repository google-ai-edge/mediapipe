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

import logging
import os
from typing import Any, Callable, Optional
import zipfile

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
from mediapipe.tasks.python.metadata.metadata_writers import face_stylizer as metadata_writer

# Face detector model and face landmarks detector file names.
_FACE_DETECTOR_MODEL = 'face_detector.tflite'
_FACE_LANDMARKS_DETECTOR_MODEL = 'face_landmarks_detector.tflite'

# The mean value used in the input tensor normalization for the face stylizer
# model.
_NORM_MEAN = 0.0
_NORM_STD = 255.0


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

  def stylize(
      self, data: classification_ds.ClassificationDataset
  ) -> classification_ds.ClassificationDataset:
    """Stylizes the images represented by the input dataset.

    Args:
      data: Dataset of input images, can contain multiple images.

    Returns:
      A dataset contains the stylized images
    """
    input_dataset = data.gen_tf_dataset(preprocess=self._preprocessor)
    output_img_list = []
    for sample in input_dataset:
      image = sample[0]
      w = self._encoder(image, training=True)
      x = self._decoder({'inputs': w + self.w_avg}, training=True)
      output_batch = x['image'][-1]
      output_img_tensor = (tf.squeeze(output_batch).numpy() + 1.0) * 127.5
      output_img_list.append(output_img_tensor)

    image_ds = tf.data.Dataset.from_tensor_slices(output_img_list)

    logging.info('Stylized %s images.', len(output_img_list))

    return classification_ds.ClassificationDataset(
        dataset=image_ds,
        label_names=['stylized'],
        size=len(output_img_list),
    )

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
    """Creates the components of face stylizer."""
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
      preprocessor: Optional[Callable[..., Any]] = None,
  ):
    """Trains the face stylizer model.

    Args:
      train_data: The data for training model.
      preprocessor: The image preprocessor.
    """
    train_dataset = train_data.gen_tf_dataset(preprocess=preprocessor)

    # TODO: Support processing multiple input style images. The
    # input style images are expected to have similar style.
    # style_sample represents a tuple of (style_image, style_label).
    style_sample = next(iter(train_dataset))
    style_img = style_sample[0]

    batch_size = self._hparams.batch_size
    label_in = tf.zeros(shape=[batch_size, 0])

    style_encoding = self._encoder(style_img, training=True) + self.w_avg

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
        outputs = self._decoder({'inputs': in_latent.numpy()}, training=True)
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
        print(f'Iteration {i} loss: {style_loss.numpy()}')

        tvars = self._decoder.trainable_variables
        grads = tape.gradient(style_loss, tvars)
        optimizer.apply_gradients(list(zip(grads, tvars)))

  def export_model(self, model_name: str = 'face_stylizer.task'):
    """Converts the model to TFLite and exports as a model bundle file.

    Saves a model bundle file and metadata json file to hparams.export_dir. The
    resulting model bundle file will contain necessary models for face
    detection, face landmarks detection, and customized face stylization. Only
    the model bundle file is needed for the downstream face stylization task.
    The metadata.json file is saved only to interpret the contents of the model
    bundle file. The face detection model and face landmarks detection model are
    from https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2.task
    and the customized face stylization model is trained in this library.

    Args:
      model_name: Face stylizer model bundle file name. The full export path is
        {self._hparams.export_dir}/{model_name}.
    """
    if not tf.io.gfile.exists(self._hparams.export_dir):
      tf.io.gfile.makedirs(self._hparams.export_dir)
    model_bundle_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, 'metadata.json')

    # Create an end-to-end model by concatenating encoder and decoder
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = self._encoder(inputs, training=True)
    x = self._decoder({'inputs': x + self.w_avg}, training=True)
    x = x['image'][-1]
    # Scale the data range from [-1, 1] to [0, 1] to support running inference
    # on both CPU and GPU.
    outputs = (x + 1.0) / 2.0
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    face_stylizer_model_buffer = model_util.convert_to_tflite(
        model=model,
        quantization_config=None,
        supported_ops=(tf.lite.OpsSet.TFLITE_BUILTINS,),
        preprocess=self._preprocessor,
        allow_custom_ops=True,
    )

    face_aligner_task_file_path = constants.FACE_ALIGNER_TASK_FILES.get_path()

    with zipfile.ZipFile(face_aligner_task_file_path, 'r') as zf:
      file_list = zf.namelist()
      if _FACE_DETECTOR_MODEL not in file_list:
        raise ValueError(
            '{0} is not packed in face aligner task file'.format(
                _FACE_DETECTOR_MODEL
            )
        )
      if _FACE_LANDMARKS_DETECTOR_MODEL not in file_list:
        raise ValueError(
            '{0} is not packed in face aligner task file'.format(
                _FACE_LANDMARKS_DETECTOR_MODEL
            )
        )

      with zf.open(_FACE_DETECTOR_MODEL) as f:
        face_detector_model_buffer = f.read()

      with zf.open(_FACE_LANDMARKS_DETECTOR_MODEL) as f:
        face_landmarks_detector_model_buffer = f.read()

    writer = metadata_writer.MetadataWriter.create(
        bytearray(face_stylizer_model_buffer),
        bytearray(face_detector_model_buffer),
        bytearray(face_landmarks_detector_model_buffer),
        input_norm_mean=[_NORM_MEAN],
        input_norm_std=[_NORM_STD],
    )

    model_bundle_content, metadata_json = writer.populate()
    with open(model_bundle_file, 'wb') as f:
      f.write(model_bundle_content)
    with open(metadata_file, 'w') as f:
      f.write(metadata_json)
