# Copyright 2023 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock
import tensorflow as tf

from mediapipe.model_maker.python.vision.object_detector import detection
from official.core import config_definitions as cfg
from official.vision import configs
from official.vision.serving import detection as detection_module


class ObjectDetectorTest(tf.test.TestCase):

  @mock.patch.object(detection_module.DetectionModule, 'serve', autospec=True)
  def test_detection_module(self, mock_serve):
    mock_serve.return_value = {
        'detection_boxes': 1,
        'detection_scores': 2,
        'detection_classes': 3,
        'num_detections': 4,
    }
    model_config = configs.retinanet.RetinaNet(
        min_level=3,
        max_level=7,
        num_classes=10,
        input_size=[256, 256, 3],
        anchor=configs.retinanet.Anchor(
            num_scales=3, aspect_ratios=[0.5, 1.0, 2.0], anchor_size=3
        ),
        backbone=configs.backbones.Backbone(
            type='mobilenet', mobilenet=configs.backbones.MobileNet()
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
        detection_generator=configs.retinanet.DetectionGenerator(),
        norm_activation=configs.common.NormActivation(activation='relu6'),
    )
    task_config = configs.retinanet.RetinaNetTask(model=model_config)
    params = cfg.ExperimentConfig(
        task=task_config,
    )
    detection_instance = detection.DetectionModule(
        params=params, batch_size=1, input_image_size=[256, 256]
    )
    outputs = detection_instance.serve(0)
    expected_outputs = {
        'detection_boxes': 1,
        'detection_scores': 2,
    }
    self.assertAllEqual(outputs, expected_outputs)


if __name__ == '__main__':
  tf.test.main()
