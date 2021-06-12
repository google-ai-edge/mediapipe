# Copyright 2019 The MediaPipe Authors.
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
r"""Code to clone the github repository, download the checkpoint and generate the frozen graph.

The frozen VGGish checkpoint will be saved to `/tmp/mediapipe/vggish_new.pb`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
import tensorflow.compat.v1 as tf
from tensorflow.python.tools import freeze_graph

BASE_DIR = '/tmp/mediapipe/'


def create_vggish_frozen_graph():
  """Create the VGGish frozen graph."""
  os.system('git clone https://github.com/tensorflow/models.git')
  sys.path.append('models/research/audioset/vggish/')

  import vggish_slim
  os.system('curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt')
  ckpt_path = 'vggish_model.ckpt'

  with tf.Graph().as_default(), tf.Session() as sess:
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, ckpt_path)

    saver = tf.train.Saver(tf.all_variables())

    freeze_graph.freeze_graph_with_def_protos(
        sess.graph_def,
        saver.as_saver_def(),
        ckpt_path,
        'vggish/fc2/BiasAdd',
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph='/tmp/mediapipe/vggish_new.pb',
        clear_devices=True,
        initializer_nodes=None)
  os.system('rm -rf models/')
  os.system('rm %s' % ckpt_path)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  create_vggish_frozen_graph()


if __name__ == '__main__':
  app.run(main)
