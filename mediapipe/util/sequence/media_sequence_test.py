"""Copyright 2019 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for media_sequence.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from mediapipe.util.sequence import media_sequence as ms


class MediaSequenceTest(tf.test.TestCase):

  def test_expected_functions_are_defined(self):
    # The code from media_sequence_util is already tested, but this test ensures
    # that we actually generate the expected methods. We only test one per
    # feature and the only test is to not crash with undefined attributes. By
    # passing in a value, we also ensure that the types are correct because the
    # underlying code crashes with a type mismatch.
    example = tf.train.SequenceExample()
    # context
    ms.set_example_id(b"string", example)
    ms.set_example_dataset_name(b"string", example)
    ms.set_example_dataset_flag_string([b"overal", b"test"], example)
    ms.set_clip_media_id(b"string", example)
    ms.set_clip_alternative_media_id(b"string", example)
    ms.set_clip_encoded_media_bytes(b"string", example)
    ms.set_clip_encoded_media_start_timestamp(47, example)
    ms.set_clip_data_path(b"string", example)
    ms.set_clip_start_timestamp(47, example)
    ms.set_clip_end_timestamp(47, example)
    ms.set_clip_label_string((b"string", b"test"), example)
    ms.set_clip_label_index((47, 49), example)
    ms.set_clip_label_confidence((0.47, 0.49), example)
    ms.set_segment_start_timestamp((47, 49), example)
    ms.set_segment_start_index((47, 49), example)
    ms.set_segment_end_timestamp((47, 49), example)
    ms.set_segment_end_index((47, 49), example)
    ms.set_segment_label_index((47, 49), example)
    ms.set_segment_label_string((b"test", b"strings"), example)
    ms.set_segment_label_confidence((0.47, 0.49), example)
    ms.set_image_format(b"test", example)
    ms.set_image_channels(47, example)
    ms.set_image_colorspace(b"test", example)
    ms.set_image_height(47, example)
    ms.set_image_width(47, example)
    ms.set_image_frame_rate(0.47, example)
    ms.set_image_data_path(b"test", example)
    ms.set_forward_flow_format(b"test", example)
    ms.set_forward_flow_channels(47, example)
    ms.set_forward_flow_colorspace(b"test", example)
    ms.set_forward_flow_height(47, example)
    ms.set_forward_flow_width(47, example)
    ms.set_forward_flow_frame_rate(0.47, example)
    ms.set_class_segmentation_format(b"test", example)
    ms.set_class_segmentation_height(47, example)
    ms.set_class_segmentation_width(47, example)
    ms.set_class_segmentation_class_label_string((b"test", b"strings"), example)
    ms.set_class_segmentation_class_label_index((47, 49), example)
    ms.set_instance_segmentation_format(b"test", example)
    ms.set_instance_segmentation_height(47, example)
    ms.set_instance_segmentation_width(47, example)
    ms.set_instance_segmentation_object_class_index((47, 49), example)
    ms.set_bbox_parts((b"HEAD", b"TOE"), example)
    ms.set_context_feature_floats((47., 35.), example)
    ms.set_context_feature_bytes((b"test", b"strings"), example)
    ms.set_context_feature_ints((47, 35), example)
    # feature lists
    ms.add_image_encoded(b"test", example)
    ms.add_image_multi_encoded([b"test", b"test"], example)
    ms.add_image_timestamp(47, example)
    ms.add_forward_flow_encoded(b"test", example)
    ms.add_forward_flow_multi_encoded([b"test", b"test"], example)
    ms.add_forward_flow_timestamp(47, example)
    ms.add_bbox_ymin((0.47, 0.49), example)
    ms.add_bbox_xmin((0.47, 0.49), example)
    ms.add_bbox_ymax((0.47, 0.49), example)
    ms.add_bbox_xmax((0.47, 0.49), example)
    ms.add_bbox_point_x((0.47, 0.49), example)
    ms.add_bbox_point_y((0.47, 0.49), example)
    ms.add_bbox_3d_point_x((0.47, 0.49), example)
    ms.add_bbox_3d_point_y((0.47, 0.49), example)
    ms.add_bbox_3d_point_z((0.47, 0.49), example)
    ms.add_predicted_bbox_ymin((0.47, 0.49), example)
    ms.add_predicted_bbox_xmin((0.47, 0.49), example)
    ms.add_predicted_bbox_ymax((0.47, 0.49), example)
    ms.add_predicted_bbox_xmax((0.47, 0.49), example)
    ms.add_bbox_num_regions(47, example)
    ms.add_bbox_is_annotated(47, example)
    ms.add_bbox_is_generated((47, 49), example)
    ms.add_bbox_is_occluded((47, 49), example)
    ms.add_bbox_label_index((47, 49), example)
    ms.add_bbox_label_string((b"test", b"strings"), example)
    ms.add_bbox_label_confidence((0.47, 0.49), example)
    ms.add_bbox_class_index((47, 49), example)
    ms.add_bbox_class_string((b"test", b"strings"), example)
    ms.add_bbox_class_confidence((0.47, 0.49), example)
    ms.add_bbox_track_index((47, 49), example)
    ms.add_bbox_track_string((b"test", b"strings"), example)
    ms.add_bbox_track_confidence((0.47, 0.49), example)
    ms.add_bbox_timestamp(47, example)
    ms.add_predicted_bbox_class_index((47, 49), example)
    ms.add_predicted_bbox_class_string((b"test", b"strings"), example)
    ms.add_predicted_bbox_timestamp(47, example)
    ms.add_class_segmentation_encoded(b"test", example)
    ms.add_class_segmentation_multi_encoded([b"test", b"test"], example)
    ms.add_instance_segmentation_encoded(b"test", example)
    ms.add_instance_segmentation_multi_encoded([b"test", b"test"], example)
    ms.add_class_segmentation_timestamp(47, example)
    ms.set_bbox_embedding_dimensions_per_region((47, 49), example)
    ms.set_bbox_embedding_format(b"test", example)
    ms.add_bbox_embedding_floats((0.47, 0.49), example)
    ms.add_bbox_embedding_encoded((b"text", b"stings"), example)
    ms.add_bbox_embedding_confidence((0.47, 0.49), example)
    ms.set_text_language(b"test", example)
    ms.set_text_context_content(b"text", example)
    ms.add_text_content(b"one", example)
    ms.add_text_timestamp(47, example)
    ms.add_text_confidence(0.47, example)
    ms.add_text_duration(47, example)
    ms.add_text_token_id(47, example)
    ms.add_text_embedding((0.47, 0.49), example)

  def test_bbox_round_trip(self):
    example = tf.train.SequenceExample()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8]])
    empty_boxes = np.array([])
    ms.add_bbox(boxes, example)
    ms.add_bbox(empty_boxes, example)
    self.assertEqual(2, ms.get_bbox_size(example))
    self.assertAllClose(boxes, ms.get_bbox_at(0, example))
    self.assertTrue(ms.has_bbox(example))
    ms.clear_bbox(example)
    self.assertEqual(0, ms.get_bbox_size(example))

  def test_point_round_trip(self):
    example = tf.train.SequenceExample()
    points = np.array([[0.1, 0.2],
                       [0.5, 0.6]])
    ms.add_bbox_point(points, example)
    ms.add_bbox_point(points, example)
    self.assertEqual(2, ms.get_bbox_point_size(example))
    self.assertAllClose(points, ms.get_bbox_point_at(0, example))
    self.assertTrue(ms.has_bbox_point(example))
    ms.clear_bbox_point(example)
    self.assertEqual(0, ms.get_bbox_point_size(example))

  def test_prefixed_point_round_trip(self):
    example = tf.train.SequenceExample()
    points = np.array([[0.1, 0.2],
                       [0.5, 0.6]])
    ms.add_bbox_point(points, example, "test")
    ms.add_bbox_point(points, example, "test")
    self.assertEqual(2, ms.get_bbox_point_size(example, "test"))
    self.assertAllClose(points, ms.get_bbox_point_at(0, example, "test"))
    self.assertTrue(ms.has_bbox_point(example, "test"))
    ms.clear_bbox_point(example, "test")
    self.assertEqual(0, ms.get_bbox_point_size(example, "test"))

  def test_3d_point_round_trip(self):
    example = tf.train.SequenceExample()
    points = np.array([[0.1, 0.2, 0.3],
                       [0.5, 0.6, 0.7]])
    ms.add_bbox_3d_point(points, example)
    ms.add_bbox_3d_point(points, example)
    self.assertEqual(2, ms.get_bbox_3d_point_size(example))
    self.assertAllClose(points, ms.get_bbox_3d_point_at(0, example))
    self.assertTrue(ms.has_bbox_3d_point(example))
    ms.clear_bbox_3d_point(example)
    self.assertEqual(0, ms.get_bbox_3d_point_size(example))

  def test_predicted_bbox_round_trip(self):
    example = tf.train.SequenceExample()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8]])
    ms.add_predicted_bbox(boxes, example)
    ms.add_predicted_bbox(boxes, example)
    self.assertEqual(2, ms.get_predicted_bbox_size(example))
    self.assertAllClose(boxes, ms.get_predicted_bbox_at(0, example))
    self.assertTrue(ms.has_predicted_bbox(example))
    ms.clear_predicted_bbox(example)
    self.assertEqual(0, ms.get_predicted_bbox_size(example))

  def test_float_list_round_trip(self):
    example = tf.train.SequenceExample()
    values_1 = [0.1, 0.2, 0.3]
    values_2 = [0.2, 0.3, 0.4]
    ms.add_feature_floats(values_1, example, "1")
    ms.add_feature_floats(values_1, example, "1")
    ms.add_feature_floats(values_2, example, "2")
    self.assertEqual(2, ms.get_feature_floats_size(example, "1"))
    self.assertEqual(1, ms.get_feature_floats_size(example, "2"))
    self.assertTrue(ms.has_feature_floats(example, "1"))
    self.assertTrue(ms.has_feature_floats(example, "2"))
    self.assertAllClose(values_1, ms.get_feature_floats_at(0, example, "1"))
    self.assertAllClose(values_2, ms.get_feature_floats_at(0, example, "2"))
    ms.clear_feature_floats(example, "1")
    self.assertEqual(0, ms.get_feature_floats_size(example, "1"))
    self.assertFalse(ms.has_feature_floats(example, "1"))
    self.assertEqual(1, ms.get_feature_floats_size(example, "2"))
    self.assertTrue(ms.has_feature_floats(example, "2"))
    ms.clear_feature_floats(example, "2")
    self.assertEqual(0, ms.get_feature_floats_size(example, "2"))
    self.assertFalse(ms.has_feature_floats(example, "2"))

  def test_feature_timestamp_round_trip(self):
    example = tf.train.SequenceExample()
    values_1 = 47
    values_2 = 49
    ms.add_feature_timestamp(values_1, example, "1")
    ms.add_feature_timestamp(values_1, example, "1")
    ms.add_feature_timestamp(values_2, example, "2")
    self.assertEqual(2, ms.get_feature_timestamp_size(example, "1"))
    self.assertEqual(1, ms.get_feature_timestamp_size(example, "2"))
    self.assertTrue(ms.has_feature_timestamp(example, "1"))
    self.assertTrue(ms.has_feature_timestamp(example, "2"))
    self.assertAllClose(values_1,
                        ms.get_feature_timestamp_at(0, example, "1"))
    self.assertAllClose(values_2,
                        ms.get_feature_timestamp_at(0, example, "2"))
    ms.clear_feature_timestamp(example, "1")
    self.assertEqual(0, ms.get_feature_timestamp_size(example, "1"))
    self.assertFalse(ms.has_feature_timestamp(example, "1"))
    self.assertEqual(1, ms.get_feature_timestamp_size(example, "2"))
    self.assertTrue(ms.has_feature_timestamp(example, "2"))
    ms.clear_feature_timestamp(example, "2")
    self.assertEqual(0, ms.get_feature_timestamp_size(example, "2"))
    self.assertFalse(ms.has_feature_timestamp(example, "2"))

  def test_feature_dimensions_round_trip(self):
    example = tf.train.SequenceExample()
    ms.set_feature_dimensions([47, 49], example, "1")
    ms.set_feature_dimensions([49, 50], example, "2")
    self.assertSequenceEqual([47, 49],
                             ms.get_feature_dimensions(example, "1"))
    self.assertSequenceEqual([49, 50],
                             ms.get_feature_dimensions(example, "2"))
    self.assertTrue(ms.has_feature_dimensions(example, "1"))
    self.assertTrue(ms.has_feature_dimensions(example, "2"))
    ms.clear_feature_dimensions(example, "1")
    self.assertFalse(ms.has_feature_dimensions(example, "1"))
    self.assertTrue(ms.has_feature_dimensions(example, "2"))
    ms.clear_feature_dimensions(example, "2")
    self.assertFalse(ms.has_feature_dimensions(example, "1"))
    self.assertFalse(ms.has_feature_dimensions(example, "2"))


if __name__ == "__main__":
  tf.test.main()
