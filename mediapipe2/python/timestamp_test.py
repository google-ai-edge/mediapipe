# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mediapipe.python._framework_bindings.timestamp."""

import time

from absl.testing import absltest
import mediapipe as mp


class TimestampTest(absltest.TestCase):

  def test_timestamp(self):
    t = mp.Timestamp(100)
    self.assertEqual(t.value, 100)
    self.assertEqual(t, 100)
    self.assertEqual(str(t), '<mediapipe.Timestamp with value: 100>')

  def test_timestamp_copy_constructor(self):
    ts1 = mp.Timestamp(100)
    ts2 = mp.Timestamp(ts1)
    self.assertEqual(ts1, ts2)

  def test_timestamp_comparsion(self):
    ts1 = mp.Timestamp(100)
    ts2 = mp.Timestamp(100)
    self.assertEqual(ts1, ts2)
    ts3 = mp.Timestamp(200)
    self.assertNotEqual(ts1, ts3)

  def test_timestamp_special_values(self):
    t1 = mp.Timestamp.UNSET
    self.assertEqual(str(t1), '<mediapipe.Timestamp with value: UNSET>')
    t2 = mp.Timestamp.UNSTARTED
    self.assertEqual(str(t2), '<mediapipe.Timestamp with value: UNSTARTED>')
    t3 = mp.Timestamp.PRESTREAM
    self.assertEqual(str(t3), '<mediapipe.Timestamp with value: PRESTREAM>')
    t4 = mp.Timestamp.MIN
    self.assertEqual(str(t4), '<mediapipe.Timestamp with value: MIN>')
    t5 = mp.Timestamp.MAX
    self.assertEqual(str(t5), '<mediapipe.Timestamp with value: MAX>')
    t6 = mp.Timestamp.POSTSTREAM
    self.assertEqual(str(t6), '<mediapipe.Timestamp with value: POSTSTREAM>')
    t7 = mp.Timestamp.DONE
    self.assertEqual(str(t7), '<mediapipe.Timestamp with value: DONE>')

  def test_timestamp_comparisons(self):
    ts1 = mp.Timestamp(100)
    ts2 = mp.Timestamp(101)
    self.assertGreater(ts2, ts1)
    self.assertGreaterEqual(ts2, ts1)
    self.assertLess(ts1, ts2)
    self.assertLessEqual(ts1, ts2)
    self.assertNotEqual(ts1, ts2)

  def test_from_seconds(self):
    now = time.time()
    ts = mp.Timestamp.from_seconds(now)
    self.assertAlmostEqual(now, ts.seconds(), delta=1)


if __name__ == '__main__':
  absltest.main()
