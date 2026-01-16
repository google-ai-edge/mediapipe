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
"""Custom Model for Object Detection."""

import os
from typing import Mapping, Optional, Sequence, Union

import tensorflow as tf

from mediapipe.model_maker.python.vision.object_detector import detection
from mediapipe.model_maker.python.vision.object_detector import model_options as model_opt
from mediapipe.model_maker.python.vision.object_detector import model_spec as ms
from official.core import config_definitions as cfg
from official.projects.qat.vision.configs import common as qat_common
from official.projects.qat.vision.modeling import factory as qat_factory
from official.vision import configs
from official.vision.losses import focal_loss
from official.vision.losses import loss_utils
from official.vision.modeling import factory
from official.vision.modeling import retinanet_model
from official.vision.modeling.layers import detection_generator


class ObjectDetectorModel(tf.keras.Model):
  """An object detector model which can be trained using Model Maker's training API.

  Attributes:
    loss_trackers: List of tf.keras.metrics.Mean objects used to track the loss
      during training.
  """

  def __init__(
      self,
      model_spec: ms.ModelSpec,
      model_options: model_opt.ObjectDetectorModelOptions,
      num_classes: int,
  ) -> None:
    """Initializes an ObjectDetectorModel.

    Args:
      model_spec: Specification for the model.
      model_options: Model options for creating the model.
      num_classes: Number of classes for object detection.
    """
    super().__init__()
    self._model_spec = model_spec
    self._model_options = model_options
    self._num_classes = num_classes
    self._model = self._build_model()
    checkpoint_folder = self._model_spec.downloaded_files.get_path()
    checkpoint_file = os.path.join(
        checkpoint_folder, self._model_spec.checkpoint_name
    )
    self.load_checkpoint(checkpoint_file)
    self._model.summary()
    self.loss_trackers = [
        tf.keras.metrics.Mean(name=n)
        for n in ['total_loss', 'cls_loss', 'box_loss', 'model_loss']
    ]

  def _get_model_config(
      self,
      generator_config: configs.retinanet.DetectionGenerator = configs.retinanet.DetectionGenerator(),
  ) -> configs.retinanet.RetinaNet:
    model_config = configs.retinanet.RetinaNet(
        min_level=self._model_spec.min_level,
        max_level=self._model_spec.max_level,
        num_classes=self._num_classes,
        input_size=self._model_spec.input_image_shape,
        anchor=configs.retinanet.Anchor(
            num_scales=3, aspect_ratios=[0.5, 1.0, 2.0], anchor_size=3
        ),
        backbone=configs.backbones.Backbone(
            type='mobilenet',
            mobilenet=configs.backbones.MobileNet(
                model_id=self._model_spec.model_id
            ),
        ),
        decoder=configs.decoders.Decoder(
            type='fpn',
            fpn=configs.decoders.FPN(
                num_filters=128, use_separable_conv=True, use_keras_layer=True
            ),
        ),
        head=configs.retinanet.RetinaNetHead(
            num_filters=128, use_separable_conv=True
        ),
        detection_generator=generator_config,
        norm_activation=configs.common.NormActivation(activation='relu6'),
    )
    return model_config

  def _build_model(self, omit_l2=False) -> tf.keras.Model:
    """Builds a RetinaNet object detector model."""
    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self._model_spec.input_image_shape
    )
    if omit_l2:
      l2_regularizer = None
    else:
      l2_regularizer = tf.keras.regularizers.l2(
          self._model_options.l2_weight_decay / 2.0
      )
    model_config = self._get_model_config()

    return factory.build_retinanet(input_specs, model_config, l2_regularizer)

  def save_checkpoint(self, checkpoint_path: str) -> None:
    """Saves a model checkpoint to checkpoint_path.

    Args:
      checkpoint_path: The path to save checkpoint.
    """
    ckpt_items = {
        'backbone': self._model.backbone,
        'decoder': self._model.decoder,
        'head': self._model.head,
    }
    tf.train.Checkpoint(**ckpt_items).write(checkpoint_path)

  def load_checkpoint(
      self, checkpoint_path: str, include_last_layer: bool = False
  ) -> None:
    """Loads a model checkpoint from checkpoint_path.

    Args:
      checkpoint_path: The path to load a checkpoint from.
      include_last_layer: Whether or not to load the last classification layer.
        The size of the last classification layer will differ depending on the
        number of classes. When loading from the pre-trained checkpoint, this
        parameter should be False to avoid shape mismatch on the last layer.
        Defaults to False.
    """
    dummy_input = tf.zeros([1] + self._model_spec.input_image_shape)
    self._model(dummy_input, training=True)
    if include_last_layer:
      head = self._model.head
    else:
      head_classifier = tf.train.Checkpoint(
          depthwise_kernel=self._model.head._classifier.depthwise_kernel  # pylint:disable=protected-access
      )
      head_items = {
          '_classifier': head_classifier,
          '_box_norms': self._model.head._box_norms,  # pylint:disable=protected-access
          '_box_regressor': self._model.head._box_regressor,  # pylint:disable=protected-access
          '_cls_convs': self._model.head._cls_convs,  # pylint:disable=protected-access
          '_cls_norms': self._model.head._cls_norms,  # pylint:disable=protected-access
          '_box_convs': self._model.head._box_convs,  # pylint:disable=protected-access
      }
      head = tf.train.Checkpoint(**head_items)
    ckpt_items = {
        'backbone': self._model.backbone,
        'decoder': self._model.decoder,
        'head': head,
    }
    ckpt = tf.train.Checkpoint(**ckpt_items)
    status = ckpt.read(checkpoint_path)
    status.expect_partial().assert_existing_objects_matched()

  def convert_to_qat(self) -> None:
    """Converts the model to a QAT RetinaNet model."""
    model = self._build_model(omit_l2=True)
    dummy_input = tf.zeros([1] + self._model_spec.input_image_shape)
    model(dummy_input, training=True)
    model.set_weights(self._model.get_weights())
    quantization_config = qat_common.Quantization(
        quantize_detection_decoder=True, quantize_detection_head=True
    )
    model_config = self._get_model_config()
    qat_model = qat_factory.build_qat_retinanet(
        model, quantization_config, model_config
    )
    self._model = qat_model

  def export_saved_model(self, save_path: str):
    """Exports a saved_model for tflite conversion.

    The export process modifies the model in the following two ways:
      1. Replaces the nms operation in the detection generator with a custom
        TFLite compatible nms operation.
      2. Wraps the model with a DetectionModule which handles pre-processing
        and post-processing when running inference.

    Args:
      save_path: Path to export the saved model.
    """
    generator_config = configs.retinanet.DetectionGenerator(
        nms_version='tflite',
        tflite_post_processing=configs.common.TFLitePostProcessingConfig(
            nms_score_threshold=0,
            max_detections=10,
            max_classes_per_detection=1,
            normalize_anchor_coordinates=True,
            omit_nms=True,
        ),
    )
    tflite_post_processing_config = (
        generator_config.tflite_post_processing.as_dict()
    )
    tflite_post_processing_config['input_image_size'] = (
        self._model_spec.input_image_shape[0],
        self._model_spec.input_image_shape[1],
    )
    detection_generator_obj = detection_generator.MultilevelDetectionGenerator(
        apply_nms=generator_config.apply_nms,
        pre_nms_top_k=generator_config.pre_nms_top_k,
        pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
        nms_iou_threshold=generator_config.nms_iou_threshold,
        max_num_detections=generator_config.max_num_detections,
        nms_version=generator_config.nms_version,
        use_cpu_nms=generator_config.use_cpu_nms,
        soft_nms_sigma=generator_config.soft_nms_sigma,
        tflite_post_processing_config=tflite_post_processing_config,
        return_decoded=generator_config.return_decoded,
        use_class_agnostic_nms=generator_config.use_class_agnostic_nms,
    )
    model_config = self._get_model_config(generator_config)
    model = retinanet_model.RetinaNetModel(
        self._model.backbone,
        self._model.decoder,
        self._model.head,
        detection_generator_obj,
        min_level=model_config.min_level,
        max_level=model_config.max_level,
        num_scales=model_config.anchor.num_scales,
        aspect_ratios=model_config.anchor.aspect_ratios,
        anchor_size=model_config.anchor.anchor_size,
    )
    task_config = configs.retinanet.RetinaNetTask(model=model_config)
    params = cfg.ExperimentConfig(
        task=task_config,
    )
    export_module = detection.DetectionModule(
        params=params,
        batch_size=1,
        input_image_size=self._model_spec.input_image_shape[:2],
        input_type='tflite',
        num_channels=self._model_spec.input_image_shape[2],
        model=model,
    )
    function_keys = {'tflite': tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY}
    signatures = export_module.get_inference_signatures(function_keys)

    tf.saved_model.save(export_module, save_path, signatures=signatures)

  # The remaining method overrides are used to train this object detector model
  # using model.fit().
  def call(  # pytype: disable=annotation-type-mismatch
      self,
      images: Union[tf.Tensor, Sequence[tf.Tensor]],
      image_shape: Optional[tf.Tensor] = None,
      anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
      output_intermediate_features: bool = False,
      training: bool = None,
  ) -> Mapping[str, tf.Tensor]:
    """Overrides call from tf.keras.Model."""
    return self._model(
        images,
        image_shape,
        anchor_boxes,
        output_intermediate_features,
        training,
    )

  def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
    """Overrides compute_loss from tf.keras.Model."""
    cls_loss_fn = focal_loss.FocalLoss(
        alpha=0.25, gamma=1.5, reduction=tf.keras.losses.Reduction.SUM
    )
    box_loss_fn = tf.keras.losses.Huber(
        0.1, reduction=tf.keras.losses.Reduction.SUM
    )
    labels = y
    outputs = y_pred
    # Sums all positives in a batch for normalization and avoids zero
    # num_positives_sum, which would lead to inf loss during training
    cls_sample_weight = labels['cls_weights']
    box_sample_weight = labels['box_weights']
    num_positives = tf.reduce_sum(box_sample_weight) + 1.0
    cls_sample_weight = cls_sample_weight / num_positives
    box_sample_weight = box_sample_weight / num_positives
    y_true_cls = loss_utils.multi_level_flatten(
        labels['cls_targets'], last_dim=None
    )
    y_true_cls = tf.one_hot(y_true_cls, self._num_classes)
    y_pred_cls = loss_utils.multi_level_flatten(
        outputs['cls_outputs'], last_dim=self._num_classes
    )
    y_true_box = loss_utils.multi_level_flatten(
        labels['box_targets'], last_dim=4
    )
    y_pred_box = loss_utils.multi_level_flatten(
        outputs['box_outputs'], last_dim=4
    )

    cls_loss = cls_loss_fn(
        y_true=y_true_cls, y_pred=y_pred_cls, sample_weight=cls_sample_weight
    )
    box_loss = box_loss_fn(
        y_true=y_true_box, y_pred=y_pred_box, sample_weight=box_sample_weight
    )

    model_loss = cls_loss + 50 * box_loss
    total_loss = model_loss
    regularization_losses = self._model.losses
    if regularization_losses:
      reg_loss = tf.reduce_sum(regularization_losses)
      total_loss = model_loss + reg_loss
    all_losses = {
        'total_loss': total_loss,
        'cls_loss': cls_loss,
        'box_loss': box_loss,
        'model_loss': model_loss,
    }
    for m in self.metrics:
      m.update_state(all_losses[m.name])
    return total_loss

  @property
  def metrics(self):
    """Overrides metrics from tf.keras.Model."""
    return self.loss_trackers

  def compute_metrics(self, x, y, y_pred, sample_weight=None):
    """Overrides compute_metrics from tf.keras.Model."""
    return self.get_metrics_result()

  def train_step(self, data):
    """Overrides train_step from tf.keras.Model."""
    tf.keras.backend.set_learning_phase(1)
    x, y = data
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compute_loss(x, y, y_pred)
    self._validate_target_and_loss(y, loss)
    # Run backwards pass.
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    return self.compute_metrics(x, y, y_pred)

  def test_step(self, data):
    """Overrides test_step from tf.keras.Model."""
    tf.keras.backend.set_learning_phase(0)
    x, y = data
    y_pred = self(
        x,
        anchor_boxes=y['anchor_boxes'],
        image_shape=y['image_info'][:, 1, :],
        training=False,
    )
    # Updates stateful loss metrics.
    self.compute_loss(x, y, y_pred)
    return self.compute_metrics(x, y, y_pred)
