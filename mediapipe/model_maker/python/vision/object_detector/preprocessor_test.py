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

import random

from absl.testing import parameterized
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import test_utils
from mediapipe.model_maker.python.vision.object_detector import model_spec as ms
from mediapipe.model_maker.python.vision.object_detector import preprocessor as preprocessor_lib


class DatasetTest(tf.test.TestCase, parameterized.TestCase):
  MAX_IMAGE_SIZE = 360
  OUTPUT_SIZE = 256
  NUM_CLASSES = 10
  NUM_EXAMPLES = 3
  MIN_LEVEL = 3
  MAX_LEVEL = 7
  NUM_SCALES = 3
  ASPECT_RATIOS = [0.5, 1, 2]
  MAX_NUM_INSTANCES = 100

  def _get_rand_example(self):
    num_annotations = random.randint(1, 3)
    bboxes, classes, is_crowds = [], [], []
    image_size = random.randint(10, self.MAX_IMAGE_SIZE + 1)
    rgb = [random.uniform(0, 255) for _ in range(3)]
    image = test_utils.fill_image(rgb, image_size)
    for _ in range(num_annotations):
      x1, x2 = random.uniform(0, image_size), random.uniform(0, image_size)
      y1, y2 = random.uniform(0, image_size), random.uniform(0, image_size)
      bbox = [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]
      bboxes.append(bbox)
      classes.append(random.randint(0, self.NUM_CLASSES - 1))
      is_crowds.append(0)
    return {
        'image': tf.cast(image, dtype=tf.float32),
        'groundtruth_boxes': tf.cast(bboxes, dtype=tf.float32),
        'groundtruth_classes': tf.cast(classes, dtype=tf.int64),
        'groundtruth_is_crowd': tf.cast(is_crowds, dtype=tf.bool),
        'groundtruth_area': tf.cast(is_crowds, dtype=tf.float32),
        'source_id': tf.cast(1, dtype=tf.int64),
        'height': tf.cast(image_size, dtype=tf.int64),
        'width': tf.cast(image_size, dtype=tf.int64),
    }

  def setUp(self):
    super().setUp()
    dataset = [self._get_rand_example() for _ in range(self.NUM_EXAMPLES)]

    def my_generator(data):
      for item in data:
        yield item

    self.dataset = tf.data.Dataset.from_generator(
        lambda: my_generator(dataset),
        output_types={
            'image': tf.float32,
            'groundtruth_classes': tf.int64,
            'groundtruth_boxes': tf.float32,
            'groundtruth_is_crowd': tf.bool,
            'groundtruth_area': tf.float32,
            'source_id': tf.int64,
            'height': tf.int64,
            'width': tf.int64,
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='training',
          is_training=True,
      ),
      dict(
          testcase_name='evaluation',
          is_training=False,
      ),
  )
  def test_preprocessor(self, is_training):
    model_spec = ms.SupportedModels.MOBILENET_V2.value()
    labels_keys = [
        'cls_targets',
        'box_targets',
        'anchor_boxes',
        'cls_weights',
        'box_weights',
        'image_info',
    ]
    if not is_training:
      labels_keys.append('groundtruths')
    preprocessor = preprocessor_lib.Preprocessor(model_spec)
    for example in self.dataset:
      result = preprocessor(example, is_training=is_training)
      image, labels = result
      self.assertAllEqual(image.shape, (256, 256, 3))
      self.assertCountEqual(labels_keys, labels.keys())
      np_labels = tf.nest.map_structure(lambda x: x.numpy(), labels)
      # Checks shapes of `image_info` and `anchor_boxes`.
      self.assertEqual(np_labels['image_info'].shape, (4, 2))
      n_anchors = 0
      for level in range(self.MIN_LEVEL, self.MAX_LEVEL + 1):
        stride = 2**level
        output_size_l = [self.OUTPUT_SIZE / stride, self.OUTPUT_SIZE / stride]
        anchors_per_location = self.NUM_SCALES * len(self.ASPECT_RATIOS)
        self.assertEqual(
            list(np_labels['anchor_boxes'][str(level)].shape),
            [output_size_l[0], output_size_l[1], 4 * anchors_per_location],
        )
        n_anchors += output_size_l[0] * output_size_l[1] * anchors_per_location
      # Checks shapes of training objectives.
      self.assertEqual(np_labels['cls_weights'].shape, (int(n_anchors),))
      for level in range(self.MIN_LEVEL, self.MAX_LEVEL + 1):
        stride = 2**level
        output_size_l = [self.OUTPUT_SIZE / stride, self.OUTPUT_SIZE / stride]
        anchors_per_location = self.NUM_SCALES * len(self.ASPECT_RATIOS)
        self.assertEqual(
            list(np_labels['cls_targets'][str(level)].shape),
            [output_size_l[0], output_size_l[1], anchors_per_location],
        )
        self.assertEqual(
            list(np_labels['box_targets'][str(level)].shape),
            [output_size_l[0], output_size_l[1], 4 * anchors_per_location],
        )
      # Checks shape of groundtruths for eval.
      if not is_training:
        self.assertEqual(np_labels['groundtruths']['source_id'].shape, ())
        self.assertEqual(
            np_labels['groundtruths']['classes'].shape,
            (self.MAX_NUM_INSTANCES,),
        )
        self.assertEqual(
            np_labels['groundtruths']['boxes'].shape,
            (self.MAX_NUM_INSTANCES, 4),
        )
        self.assertEqual(
            np_labels['groundtruths']['areas'].shape, (self.MAX_NUM_INSTANCES,)
        )
        self.assertEqual(
            np_labels['groundtruths']['is_crowds'].shape,
            (self.MAX_NUM_INSTANCES,),
        )


if __name__ == '__main__':
  tf.test.main()
