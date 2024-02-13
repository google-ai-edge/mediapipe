# Copyright 2024 The MediaPipe Authors.
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

"""Unit tests for pax_converter."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from mediapipe.tasks.python.genai.converter import weight_bins_writer


class WeightBinsWriterTest(parameterized.TestCase):

  def test_get_weight_info(self):
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    writer = weight_bins_writer.WeightBinsWriter(
        output_dir=output_dir, backend='cpu'
    )
    var_name = 'params.lm.softmax.logits_ffn.linear.w'
    weight_info = writer.get_weight_info(
        var_name, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    )
    self.assertEqual(
        weight_info,
        'mdl_vars.params.lm.softmax.logits_ffn.linear.w.float32.2_3\n',
    )

  def test_load_to_actions(self):
    output_dir = os.path.join(flags.FLAGS.test_tmpdir, 'output_dir')
    writer = weight_bins_writer.WeightBinsWriter(
        output_dir=output_dir, backend='cpu'
    )
    variables = {
        'mdl_vars.params.lm.softmax.logits_ffn.linear.w': (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            False,
        ),
    }
    writer.write_variables(variables)
    file_size = os.path.getsize(
        os.path.join(output_dir, 'params.lm.softmax.logits_ffn.linear.w')
    )
    self.assertEqual(file_size, 6 * 4)


if __name__ == '__main__':
  absltest.main()
