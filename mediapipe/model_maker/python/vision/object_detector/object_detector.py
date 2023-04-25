# Copyright 2023 The MediaPipe Authors.
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
"""APIs to train object detector model."""
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from mediapipe.model_maker.python.core.tasks import classifier
from mediapipe.model_maker.python.core.utils import model_util
from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.vision.object_detector import dataset as ds
from mediapipe.model_maker.python.vision.object_detector import hyperparameters as hp
from mediapipe.model_maker.python.vision.object_detector import model as model_lib
from mediapipe.model_maker.python.vision.object_detector import model_options as model_opt
from mediapipe.model_maker.python.vision.object_detector import model_spec as ms
from mediapipe.model_maker.python.vision.object_detector import object_detector_options
from mediapipe.model_maker.python.vision.object_detector import preprocessor
from mediapipe.tasks.python.metadata.metadata_writers import metadata_writer
from mediapipe.tasks.python.metadata.metadata_writers import object_detector as object_detector_writer
from official.vision.evaluation import coco_evaluator


class ObjectDetector(classifier.Classifier):
  """ObjectDetector for building object detection model."""

  def __init__(
      self,
      model_spec: ms.ModelSpec,
      label_names: List[str],
      hparams: hp.HParams,
      model_options: model_opt.ObjectDetectorModelOptions,
  ) -> None:
    """Initializes ObjectDetector class.

    Args:
      model_spec: Specifications for the model.
      label_names: A list of label names for the classes.
      hparams: The hyperparameters for training object detector.
      model_options: Options for creating the object detector model.
    """
    super().__init__(
        model_spec=model_spec, label_names=label_names, shuffle=hparams.shuffle
    )
    self._preprocessor = preprocessor.Preprocessor(model_spec)
    self._hparams = hparams
    self._model_options = model_options
    self._is_qat = False

  @classmethod
  def create(
      cls,
      train_data: ds.Dataset,
      validation_data: ds.Dataset,
      options: object_detector_options.ObjectDetectorOptions,
  ) -> 'ObjectDetector':
    """Creates and trains an ObjectDetector.

    Loads data and trains the model based on data for object detection.

    Args:
      train_data: Training data.
      validation_data: Validation data.
      options: Configurations for creating and training object detector.

    Returns:
      An instance of ObjectDetector.
    """
    if options.hparams is None:
      options.hparams = hp.HParams()

    if options.model_options is None:
      options.model_options = model_opt.ObjectDetectorModelOptions()

    spec = ms.SupportedModels.get(options.supported_model)
    object_detector = cls(
        model_spec=spec,
        label_names=train_data.label_names,
        hparams=options.hparams,
        model_options=options.model_options,
    )
    object_detector._create_and_train_model(train_data, validation_data)
    return object_detector

  def _create_and_train_model(
      self, train_data: ds.Dataset, validation_data: ds.Dataset
  ):
    """Creates and trains the model.

    Args:
      train_data: Training data.
      validation_data: Validation data.
    """
    self._optimizer = self._create_optimizer(
        model_util.get_steps_per_epoch(
            steps_per_epoch=self._hparams.steps_per_epoch,
            batch_size=self._hparams.batch_size,
            train_data=train_data,
        )
    )
    self._create_model()
    self._train_model(
        train_data, validation_data, preprocessor=self._preprocessor
    )
    self._save_float_ckpt()

  def _create_model(self) -> None:
    """Creates the object detector model."""
    self._model = model_lib.ObjectDetectorModel(
        model_spec=self._model_spec,
        model_options=self._model_options,
        num_classes=self._num_classes,
    )

  def _save_float_ckpt(self) -> None:
    """Saves a checkpoint of the trained float model.

    The default save path is {hparams.export_dir}/float_ckpt. Note that
      `float_cpt` represents a file prefix, not directory. The resulting files
      saved to {hparams.export_dir} will be:
        - float_ckpt.data-00000-of-00001
        - float_ckpt.index
    """
    save_path = os.path.join(self._hparams.export_dir, 'float_ckpt')
    if not os.path.exists(self._hparams.export_dir):
      os.makedirs(self._hparams.export_dir)
    self._model.save_checkpoint(save_path)

  def restore_float_ckpt(self) -> None:
    """Loads a float checkpoint of the model from {hparams.export_dir}/float_ckpt.

    The float checkpoint at {hparams.export_dir}/float_ckpt is automatically
    saved after training an ObjectDetector using the `create` method. This
    method is used to restore the trained float checkpoint state of the model in
    order to run `quantization_aware_training` multiple times. Example usage:

    # Train a model
    model = object_detector.create(...)
    # Run QAT
    model.quantization_aware_training(...)
    model.evaluate(...)
    # Restore the float checkpoint to run QAT again
    model.restore_float_ckpt()
    # Run QAT with different parameters
    model.quantization_aware_training(...)
    model.evaluate(...)
    """
    self._create_model()
    self._model.load_checkpoint(
        os.path.join(self._hparams.export_dir, 'float_ckpt'),
        include_last_layer=True,
    )
    self._model.compile()
    self._is_qat = False

  # TODO: Refactor this method to utilize shared training function
  def quantization_aware_training(
      self,
      train_data: ds.Dataset,
      validation_data: ds.Dataset,
      qat_hparams: hp.QATHParams,
  ) -> None:
    """Runs quantization aware training(QAT) on the model.

    The QAT step happens after training a regular float model from the `create`
    method. This additional step will fine-tune the model with a lower precision
    in order mimic the behavior of a quantized model. The resulting quantized
    model generally has better performance than a model which is quantized
    without running QAT. See the following link for more information:
    - https://www.tensorflow.org/model_optimization/guide/quantization/training

    Just like training the float model using the `create` method, the QAT step
    also requires some manual tuning of hyperparameters. In order to run QAT
    more than once for purposes such as hyperparameter tuning, use the
    `restore_float_ckpt` method to restore the model state to the trained float
    checkpoint without having to rerun the `create` method.

    Args:
      train_data: Training dataset.
      validation_data: Validaiton dataset.
      qat_hparams: Configuration for QAT.
    """
    self._model.convert_to_qat()
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        qat_hparams.learning_rate * qat_hparams.batch_size / 256,
        decay_steps=qat_hparams.decay_steps,
        decay_rate=qat_hparams.decay_rate,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=learning_rate_fn, momentum=0.9
    )
    if len(train_data) < qat_hparams.batch_size:
      raise ValueError(
          f"The size of the train_data {len(train_data)} can't be smaller than"
          f' batch_size {qat_hparams.batch_size}. To solve this problem, set'
          ' the batch_size smaller or increase the size of the train_data.'
      )

    train_dataset = train_data.gen_tf_dataset(
        batch_size=qat_hparams.batch_size,
        is_training=True,
        shuffle=self._shuffle,
        preprocess=self._preprocessor,
    )
    steps_per_epoch = model_util.get_steps_per_epoch(
        steps_per_epoch=None,
        batch_size=qat_hparams.batch_size,
        train_data=train_data,
    )
    train_dataset = train_dataset.take(count=steps_per_epoch)
    validation_dataset = validation_data.gen_tf_dataset(
        batch_size=qat_hparams.batch_size,
        is_training=False,
        preprocess=self._preprocessor,
    )
    self._model.compile(optimizer=optimizer)
    self._model.fit(
        x=train_dataset,
        epochs=qat_hparams.epochs,
        steps_per_epoch=None,
        validation_data=validation_dataset,
    )
    self._is_qat = True

  def evaluate(
      self, dataset: ds.Dataset, batch_size: int = 1
  ) -> Tuple[List[float], Dict[str, float]]:
    """Overrides Classifier.evaluate to calculate COCO metrics."""
    dataset = dataset.gen_tf_dataset(
        batch_size, is_training=False, preprocess=self._preprocessor
    )
    losses = self._model.evaluate(dataset)
    coco_eval = coco_evaluator.COCOEvaluator(
        annotation_file=None,
        include_mask=False,
        per_category_metrics=True,
        max_num_eval_detections=100,
    )
    for batch in dataset:
      x, y = batch
      y_pred = self._model(
          x,
          anchor_boxes=y['anchor_boxes'],
          image_shape=y['image_info'][:, 1, :],
          training=False,
      )
      groundtruths = y['groundtruths']
      y_pred['image_info'] = groundtruths['image_info']
      y_pred['source_id'] = groundtruths['source_id']
      coco_eval.update_state(groundtruths, y_pred)
    coco_metrics = coco_eval.result()
    return losses, coco_metrics

  def export_model(
      self,
      model_name: str = 'model.tflite',
      quantization_config: Optional[quantization.QuantizationConfig] = None,
  ):
    """Converts and saves the model to a TFLite file with metadata included.

    The model export format is automatically set based on whether or not
    `quantization_aware_training`(QAT) was run. The model exports to float32 by
    default and will export to an int8 quantized model if QAT was run. To export
    a float32 model after running QAT, run `restore_float_ckpt` before this
    method. For custom post-training quantization without QAT, use the
    quantization_config parameter.

    Note that only the TFLite file is needed for deployment. This function also
    saves a metadata.json file to the same directory as the TFLite file which
    can be used to interpret the metadata content in the TFLite file.

    Args:
      model_name: File name to save TFLite model with metadata. The full export
        path is {self._hparams.export_dir}/{model_name}.
      quantization_config: The configuration for model quantization. Note that
        int8 quantization aware training is automatically applied when possible.
        This parameter is used to specify other post-training quantization
        options such as fp16 and int8 without QAT.

    Raises:
      ValueError: If a custom quantization_config is specified when the model
        has quantization aware training enabled.
    """
    if quantization_config:
      if self._is_qat:
        raise ValueError(
            'Exporting a qat model with a custom quantization_config is not '
            'supported.'
        )
      else:
        print(
            'Exporting with custom post-training-quantization: ',
            quantization_config,
        )
    else:
      if self._is_qat:
        print('Exporting a qat int8 model')
        quantization_config = quantization.QuantizationConfig(
            inference_input_type=tf.uint8, inference_output_type=tf.uint8
        )
      else:
        print('Exporting a floating point model')

    tflite_file = os.path.join(self._hparams.export_dir, model_name)
    metadata_file = os.path.join(self._hparams.export_dir, 'metadata.json')
    with tempfile.TemporaryDirectory() as temp_dir:
      save_path = os.path.join(temp_dir, 'saved_model')
      self._model.export_saved_model(save_path)
      converter = tf.lite.TFLiteConverter.from_saved_model(save_path)
      if quantization_config:
        converter = quantization_config.set_converter_with_quantization(
            converter, preprocess=self._preprocessor
        )

      converter.target_spec.supported_ops = (tf.lite.OpsSet.TFLITE_BUILTINS,)
      tflite_model = converter.convert()

    writer = object_detector_writer.MetadataWriter.create(
        tflite_model,
        self._model_spec.mean_rgb,
        self._model_spec.stddev_rgb,
        labels=metadata_writer.Labels().add(list(self._label_names)),
    )
    tflite_model_with_metadata, metadata_json = writer.populate()
    model_util.save_tflite(tflite_model_with_metadata, tflite_file)
    with open(metadata_file, 'w') as f:
      f.write(metadata_json)

  def _create_optimizer(
      self, steps_per_epoch: int
  ) -> tf.keras.optimizers.Optimizer:
    """Creates an optimizer with learning rate schedule for regular training.

    Uses Keras PiecewiseConstantDecay schedule by default.

    Args:
      steps_per_epoch: Steps per epoch to calculate the step boundaries from the
        learning_rate_epoch_boundaries

    Returns:
      A tf.keras.optimizer.Optimizer for model training.
    """
    init_lr = self._hparams.learning_rate * self._hparams.batch_size / 256
    if self._hparams.learning_rate_epoch_boundaries:
      lr_values = [init_lr] + [
          init_lr * m for m in self._hparams.learning_rate_decay_multipliers
      ]
      lr_step_boundaries = [
          steps_per_epoch * epoch_boundary
          for epoch_boundary in self._hparams.learning_rate_epoch_boundaries
      ]
      learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          lr_step_boundaries, lr_values
      )
    else:
      learning_rate = init_lr
    return tf.keras.optimizers.experimental.SGD(
        learning_rate=learning_rate, momentum=0.9
    )
