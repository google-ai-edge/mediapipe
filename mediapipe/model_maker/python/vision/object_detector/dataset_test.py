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

import json
import os
import random
import tensorflow as tf

from mediapipe.model_maker.python.vision.core import image_utils
from mediapipe.model_maker.python.vision.core import test_utils
from mediapipe.model_maker.python.vision.object_detector import dataset
from mediapipe.tasks.python.test import test_utils as tasks_test_utils

IMAGE_SIZE = 224


class DatasetTest(tf.test.TestCase):

  def _get_rand_bbox(self):
    x1, x2 = random.uniform(0, IMAGE_SIZE), random.uniform(0, IMAGE_SIZE)
    y1, y2 = random.uniform(0, IMAGE_SIZE), random.uniform(0, IMAGE_SIZE)
    return [min(x1, x2), min(y1, y2), abs(x1 - x2), abs(y1 - y2)]

  def setUp(self):
    super().setUp()
    self.coco_dataset_path = os.path.join(self.get_temp_dir(), 'coco_dataset')
    if os.path.exists(self.coco_dataset_path):
      return
    os.mkdir(self.coco_dataset_path)
    categories = [{'id': 1, 'name': 'daisy'}, {'id': 2, 'name': 'tulips'}]
    images = [
        {'id': 1, 'file_name': 'img1.jpeg'},
        {'id': 2, 'file_name': 'img2.jpeg'},
    ]
    annotations = [
        {'image_id': 1, 'category_id': 1, 'bbox': self._get_rand_bbox()},
        {'image_id': 2, 'category_id': 1, 'bbox': self._get_rand_bbox()},
        {'image_id': 2, 'category_id': 2, 'bbox': self._get_rand_bbox()},
    ]
    labels_dict = {
        'categories': categories,
        'images': images,
        'annotations': annotations,
    }
    labels_json = json.dumps(labels_dict)
    with open(os.path.join(self.coco_dataset_path, 'labels.json'), 'w') as f:
      f.write(labels_json)
    images_dir = os.path.join(self.coco_dataset_path, 'images')
    os.mkdir(images_dir)
    for item in images:
      test_utils.write_filled_jpeg_file(
          os.path.join(images_dir, item['file_name']),
          [random.uniform(0, 255) for _ in range(3)],
          IMAGE_SIZE,
      )

  def test_from_coco_folder(self):
    data = dataset.Dataset.from_coco_folder(
        self.coco_dataset_path, cache_dir=self.get_temp_dir()
    )
    self.assertLen(data, 2)
    self.assertEqual(data.num_classes, 3)
    self.assertEqual(data.label_names, ['background', 'daisy', 'tulips'])
    for example in data.gen_tf_dataset():
      boxes = example['groundtruth_boxes']
      classes = example['groundtruth_classes']
      self.assertNotEmpty(boxes)
      self.assertAllLessEqual(boxes, 1)
      self.assertAllGreaterEqual(boxes, 0)
      self.assertNotEmpty(classes)
      self.assertTrue(
          (classes.numpy() == [1]).all() or (classes.numpy() == [1, 2]).all()
      )
      if (classes.numpy() == [1, 1]).all():
        raw_image_tensor = image_utils.load_image(
            os.path.join(self.coco_dataset_path, 'images', 'img1.jpeg')
        )
      else:
        raw_image_tensor = image_utils.load_image(
            os.path.join(self.coco_dataset_path, 'images', 'img2.jpeg')
        )
      self.assertTrue(
          (example['image'].numpy() == raw_image_tensor.numpy()).all()
      )

  def test_from_pascal_voc_folder(self):
    pascal_voc_folder = tasks_test_utils.get_test_data_path('pascal_voc_data')
    data = dataset.Dataset.from_pascal_voc_folder(
        pascal_voc_folder, cache_dir=self.get_temp_dir()
    )
    self.assertLen(data, 4)
    self.assertEqual(data.num_classes, 3)
    self.assertEqual(data.label_names, ['background', 'android', 'pig_android'])
    for example in data.gen_tf_dataset():
      boxes = example['groundtruth_boxes']
      classes = example['groundtruth_classes']
      self.assertNotEmpty(boxes)
      self.assertAllLessEqual(boxes, 1)
      self.assertAllGreaterEqual(boxes, 0)
      self.assertNotEmpty(classes)
      image = example['image']
      self.assertNotEmpty(image)
      self.assertAllGreaterEqual(image, 0)
      self.assertAllLessEqual(image, 255)


if __name__ == '__main__':
  tf.test.main()
