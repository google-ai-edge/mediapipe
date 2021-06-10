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

Tests for media_sequence_util.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

from mediapipe.util.sequence import media_sequence_util as msu

# run this code at the module level to procedurally define functions for test
# cases.
msu.create_bytes_context_feature(
    "string_context", "string_feature", module_dict=msu.__dict__)
msu.create_float_context_feature(
    "float_context", "float_feature", module_dict=msu.__dict__)
msu.create_int_context_feature(
    "int64_context", "int64_feature", module_dict=msu.__dict__)
msu.create_bytes_list_context_feature(
    "string_list_context", "string_vector_feature", module_dict=msu.__dict__)
msu.create_float_list_context_feature(
    "float_list_context", "float_vector_feature", module_dict=msu.__dict__)
msu.create_int_list_context_feature(
    "int64_list_context", "int64_vector_feature", module_dict=msu.__dict__)
msu.create_bytes_feature_list(
    "string_feature_list", "string_feature_list", module_dict=msu.__dict__)
msu.create_float_feature_list(
    "float_feature_list", "float_feature_list", module_dict=msu.__dict__)
msu.create_int_feature_list(
    "int64_feature_list", "int64_feature_list", module_dict=msu.__dict__)
msu.create_bytes_list_feature_list(
    "string_list_feature_list", "string_list_feature_list",
    module_dict=msu.__dict__)
msu.create_float_list_feature_list(
    "float_list_feature_list", "float_list_feature_list",
    module_dict=msu.__dict__)
msu.create_int_list_feature_list(
    "int64_list_feature_list", "int64_list_feature_list",
    module_dict=msu.__dict__)


class MediaSequenceTest(tf.test.TestCase):

  def test_set_context_float(self):
    example = tf.train.SequenceExample()
    key = "test_float"
    msu.set_context_float(key, 0.1, example)
    self.assertAlmostEqual(0.1,
                           example.context.feature[key].float_list.value[0])

  def test_set_context_bytes(self):
    example = tf.train.SequenceExample()
    key = "test_string"
    msu.set_context_bytes(key, b"test", example)
    self.assertEqual(b"test",
                     example.context.feature[key].bytes_list.value[0])

  def test_set_context_int(self):
    example = tf.train.SequenceExample()
    key = "test_int"
    msu.set_context_int(key, 47, example)
    self.assertEqual(47,
                     example.context.feature[key].int64_list.value[0])

  def test_set_context_float_list(self):
    example = tf.train.SequenceExample()
    key = "test_float_list"
    msu.set_context_float_list(key, [0.0, 0.1], example)
    self.assertSequenceAlmostEqual(
        [0.0, 0.1], example.context.feature[key].float_list.value)

  def test_set_context_bytes_list(self):
    example = tf.train.SequenceExample()
    key = "test_string_list"
    msu.set_context_bytes_list(key, [b"test", b"test"], example)
    self.assertSequenceAlmostEqual(
        [b"test", b"test"], example.context.feature[key].bytes_list.value)

  def test_set_context_int_list(self):
    example = tf.train.SequenceExample()
    key = "test_int_list"
    msu.set_context_int_list(key, [0, 1], example)
    self.assertSequenceAlmostEqual(
        [0, 1], example.context.feature[key].int64_list.value)

  def test_round_trip_float_list_feature(self):
    example = tf.train.SequenceExample()
    key = "test_float_features"
    msu.add_float_list(key, [0.0, 0.1], example)
    msu.add_float_list(key, [0.1, 0.2], example)
    self.assertEqual(2, msu.get_feature_list_size(key, example))
    self.assertTrue(msu.has_feature_list(key, example))
    self.assertAllClose([0.0, 0.1], msu.get_float_list_at(key, 0, example))
    self.assertAllClose([0.1, 0.2], msu.get_float_list_at(key, 1, example))
    msu.clear_feature_list(key, example)
    self.assertEqual(0, msu.get_feature_list_size(key, example))
    self.assertFalse(msu.has_feature_list(key, example))

  def test_round_trip_bytes_list_feature(self):
    example = tf.train.SequenceExample()
    key = "test_bytes_features"
    msu.add_bytes_list(key, [b"test0", b"test1"], example)
    msu.add_bytes_list(key, [b"test1", b"test2"], example)
    self.assertEqual(2, msu.get_feature_list_size(key, example))
    self.assertTrue(msu.has_feature_list(key, example))
    self.assertAllEqual([b"test0", b"test1"],
                        msu.get_bytes_list_at(key, 0, example))
    self.assertAllEqual([b"test1", b"test2"],
                        msu.get_bytes_list_at(key, 1, example))
    msu.clear_feature_list(key, example)
    self.assertEqual(0, msu.get_feature_list_size(key, example))
    self.assertFalse(msu.has_feature_list(key, example))

  def test_round_trip_int_list_feature(self):
    example = tf.train.SequenceExample()
    key = "test_ints_features"
    msu.add_int_list(key, [0, 1], example)
    msu.add_int_list(key, [1, 2], example)
    self.assertEqual(2, msu.get_feature_list_size(key, example))
    self.assertTrue(msu.has_feature_list(key, example))
    self.assertAllClose([0, 1], msu.get_int_list_at(key, 0, example))
    self.assertAllClose([1, 2], msu.get_int_list_at(key, 1, example))
    msu.clear_feature_list(key, example)
    self.assertEqual(0, msu.get_feature_list_size(key, example))
    self.assertFalse(msu.has_feature_list(key, example))

  def test_round_trip_string_context(self):
    example = tf.train.SequenceExample()
    key = "string_feature"
    msu.set_string_context(b"test", example)
    self.assertEqual(b"test", msu.get_string_context(example))
    self.assertTrue(msu.has_string_context(example))
    self.assertEqual(b"test",
                     example.context.feature[key].bytes_list.value[0])
    msu.clear_string_context(example)
    self.assertFalse(msu.has_string_context(example))
    self.assertEqual("string_feature", msu.get_string_context_key())

  def test_round_trip_float_context(self):
    example = tf.train.SequenceExample()
    key = "float_feature"
    msu.set_float_context(0.47, example)
    self.assertAlmostEqual(0.47, msu.get_float_context(example))
    self.assertTrue(msu.has_float_context(example))
    self.assertAlmostEqual(0.47,
                           example.context.feature[key].float_list.value[0])
    msu.clear_float_context(example)
    self.assertFalse(msu.has_float_context(example))
    self.assertEqual("float_feature", msu.get_float_context_key())

  def test_round_trip_int_context(self):
    example = tf.train.SequenceExample()
    key = "int64_feature"
    msu.set_int64_context(47, example)
    self.assertEqual(47, msu.get_int64_context(example))
    self.assertTrue(msu.has_int64_context(example))
    self.assertEqual(47,
                     example.context.feature[key].int64_list.value[0])
    msu.clear_int64_context(example)
    self.assertFalse(msu.has_int64_context(example))
    self.assertEqual("int64_feature", msu.get_int64_context_key())

  def test_round_trip_string_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_string_list_context([b"test0", b"test1"], example)
    self.assertSequenceEqual([b"test0", b"test1"],
                             msu.get_string_list_context(example))
    self.assertTrue(msu.has_string_list_context(example))
    msu.clear_string_list_context(example)
    self.assertFalse(msu.has_string_list_context(example))
    self.assertEqual("string_vector_feature",
                     msu.get_string_list_context_key())

  def test_round_trip_float_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_float_list_context([0.47, 0.49], example)
    self.assertSequenceAlmostEqual([0.47, 0.49],
                                   msu.get_float_list_context(example))
    self.assertTrue(msu.has_float_list_context(example))
    msu.clear_float_list_context(example)
    self.assertFalse(msu.has_float_list_context(example))
    self.assertEqual("float_vector_feature", msu.get_float_list_context_key())

  def test_round_trip_int_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_int64_list_context([47, 49], example)
    self.assertSequenceEqual([47, 49], msu.get_int64_list_context(example))
    self.assertTrue(msu.has_int64_list_context(example))
    msu.clear_int64_list_context(example)
    self.assertFalse(msu.has_int64_list_context(example))
    self.assertEqual("int64_vector_feature", msu.get_int64_list_context_key())

  def test_round_trip_string_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_string_feature_list(b"test0", example)
    msu.add_string_feature_list(b"test1", example)
    self.assertEqual(b"test0", msu.get_string_feature_list_at(0, example))
    self.assertEqual(b"test1", msu.get_string_feature_list_at(1, example))
    self.assertTrue(msu.has_string_feature_list(example))
    self.assertEqual(2, msu.get_string_feature_list_size(example))
    msu.clear_string_feature_list(example)
    self.assertFalse(msu.has_string_feature_list(example))
    self.assertEqual(0, msu.get_string_feature_list_size(example))
    self.assertEqual("string_feature_list", msu.get_string_feature_list_key())

  def test_round_trip_float_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_float_feature_list(0.47, example)
    msu.add_float_feature_list(0.49, example)
    self.assertAlmostEqual(0.47, msu.get_float_feature_list_at(0, example))
    self.assertAlmostEqual(0.49, msu.get_float_feature_list_at(1, example))
    self.assertTrue(msu.has_float_feature_list(example))
    self.assertEqual(2, msu.get_float_feature_list_size(example))
    msu.clear_float_feature_list(example)
    self.assertFalse(msu.has_float_feature_list(example))
    self.assertEqual(0, msu.get_float_feature_list_size(example))
    self.assertEqual("float_feature_list", msu.get_float_feature_list_key())

  def test_round_trip_int_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_int64_feature_list(47, example)
    msu.add_int64_feature_list(49, example)
    self.assertEqual(47, msu.get_int64_feature_list_at(0, example))
    self.assertEqual(49, msu.get_int64_feature_list_at(1, example))
    self.assertTrue(msu.has_int64_feature_list(example))
    self.assertEqual(2, msu.get_int64_feature_list_size(example))
    msu.clear_int64_feature_list(example)
    self.assertFalse(msu.has_int64_feature_list(example))
    self.assertEqual(0, msu.get_int64_feature_list_size(example))
    self.assertEqual("int64_feature_list", msu.get_int64_feature_list_key())

  def test_round_trip_string_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_string_list_feature_list([b"test0", b"test1"], example)
    msu.add_string_list_feature_list([b"test1", b"test2"], example)
    self.assertSequenceEqual([b"test0", b"test1"],
                             msu.get_string_list_feature_list_at(0, example))
    self.assertSequenceEqual([b"test1", b"test2"],
                             msu.get_string_list_feature_list_at(1, example))
    self.assertTrue(msu.has_string_list_feature_list(example))
    self.assertEqual(2, msu.get_string_list_feature_list_size(example))
    msu.clear_string_list_feature_list(example)
    self.assertFalse(msu.has_string_list_feature_list(example))
    self.assertEqual(0, msu.get_string_list_feature_list_size(example))
    self.assertEqual("string_list_feature_list",
                     msu.get_string_list_feature_list_key())

  def test_round_trip_float_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_float_list_feature_list([0.47, 0.49], example)
    msu.add_float_list_feature_list([0.49, 0.50], example)
    self.assertSequenceAlmostEqual(
        [0.47, 0.49], msu.get_float_list_feature_list_at(0, example))
    self.assertSequenceAlmostEqual(
        [0.49, 0.50], msu.get_float_list_feature_list_at(1, example))
    self.assertTrue(msu.has_float_list_feature_list(example))
    self.assertEqual(2, msu.get_float_list_feature_list_size(example))
    msu.clear_float_list_feature_list(example)
    self.assertFalse(msu.has_float_list_feature_list(example))
    self.assertEqual(0, msu.get_float_list_feature_list_size(example))
    self.assertEqual("float_list_feature_list",
                     msu.get_float_list_feature_list_key())

  def test_round_trip_int_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_int64_list_feature_list([47, 49], example)
    msu.add_int64_list_feature_list([49, 50], example)
    self.assertSequenceEqual(
        [47, 49], msu.get_int64_list_feature_list_at(0, example))
    self.assertSequenceEqual(
        [49, 50], msu.get_int64_list_feature_list_at(1, example))
    self.assertTrue(msu.has_int64_list_feature_list(example))
    self.assertEqual(2, msu.get_int64_list_feature_list_size(example))
    msu.clear_int64_list_feature_list(example)
    self.assertFalse(msu.has_int64_list_feature_list(example))
    self.assertEqual(0, msu.get_int64_list_feature_list_size(example))
    self.assertEqual("int64_list_feature_list",
                     msu.get_int64_list_feature_list_key())

  def test_prefix_int64_context(self):
    example = tf.train.SequenceExample()
    msu.set_int64_context(47, example, prefix="magic")
    self.assertFalse(msu.has_int64_context(example))
    self.assertTrue(msu.has_int64_context(example, prefix="magic"))
    self.assertEqual(47, msu.get_int64_context(example, prefix="magic"))

  def test_prefix_float_context(self):
    example = tf.train.SequenceExample()
    msu.set_float_context(47., example, prefix="magic")
    self.assertFalse(msu.has_float_context(example))
    self.assertTrue(msu.has_float_context(example, prefix="magic"))
    self.assertAlmostEqual(47., msu.get_float_context(example, prefix="magic"))

  def test_prefix_string_context(self):
    example = tf.train.SequenceExample()
    msu.set_string_context(b"47", example, prefix="magic")
    self.assertFalse(msu.has_string_context(example))
    self.assertTrue(msu.has_string_context(example, prefix="magic"))
    self.assertEqual(b"47", msu.get_string_context(example, prefix="magic"))

  def test_prefix_int64_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_int64_list_context((47,), example, prefix="magic")
    self.assertFalse(msu.has_int64_list_context(example))
    self.assertTrue(msu.has_int64_list_context(example, prefix="magic"))
    self.assertEqual([47,], msu.get_int64_list_context(example, prefix="magic"))

  def test_prefix_float_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_float_list_context((47.,), example, prefix="magic")
    self.assertFalse(msu.has_float_list_context(example))
    self.assertTrue(msu.has_float_list_context(example, prefix="magic"))
    self.assertAlmostEqual([47.,],
                           msu.get_float_list_context(example, prefix="magic"))

  def test_prefix_string_list_context(self):
    example = tf.train.SequenceExample()
    msu.set_string_list_context((b"47",), example, prefix="magic")
    self.assertFalse(msu.has_string_list_context(example))
    self.assertTrue(msu.has_string_list_context(example, prefix="magic"))
    self.assertEqual([b"47",],
                     msu.get_string_list_context(example, prefix="magic"))

  def test_prefix_int64_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_int64_feature_list(47, example, prefix="magic")
    self.assertFalse(msu.has_int64_feature_list(example))
    self.assertTrue(msu.has_int64_feature_list(example, prefix="magic"))
    self.assertEqual(47,
                     msu.get_int64_feature_list_at(0, example, prefix="magic"))

  def test_prefix_float_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_float_feature_list(47., example, prefix="magic")
    self.assertFalse(msu.has_float_feature_list(example))
    self.assertTrue(msu.has_float_feature_list(example, prefix="magic"))
    self.assertAlmostEqual(
        47., msu.get_float_feature_list_at(0, example, prefix="magic"))

  def test_prefix_string_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_string_feature_list(b"47", example, prefix="magic")
    self.assertFalse(msu.has_string_feature_list(example))
    self.assertTrue(msu.has_string_feature_list(example, prefix="magic"))
    self.assertEqual(
        b"47", msu.get_string_feature_list_at(0, example, prefix="magic"))

  def test_prefix_int64_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_int64_list_feature_list((47,), example, prefix="magic")
    self.assertFalse(msu.has_int64_list_feature_list(example))
    self.assertTrue(msu.has_int64_list_feature_list(example, prefix="magic"))
    self.assertEqual(
        [47,], msu.get_int64_list_feature_list_at(0, example, prefix="magic"))

  def test_prefix_float_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_float_list_feature_list((47.,), example, prefix="magic")
    self.assertFalse(msu.has_float_list_feature_list(example))
    self.assertTrue(msu.has_float_list_feature_list(example, prefix="magic"))
    self.assertAlmostEqual(
        [47.,], msu.get_float_list_feature_list_at(0, example, prefix="magic"))

  def test_prefix_string_list_feature_list(self):
    example = tf.train.SequenceExample()
    msu.add_string_list_feature_list((b"47",), example, prefix="magic")
    self.assertFalse(msu.has_string_list_feature_list(example))
    self.assertTrue(msu.has_string_list_feature_list(example, prefix="magic"))
    self.assertEqual(
        [b"47",],
        msu.get_string_list_feature_list_at(0, example, prefix="magic"))

if __name__ == "__main__":
  tf.test.main()
