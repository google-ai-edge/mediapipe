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

"""Generate a MediaSequence metadata for MediaPipe input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import app
from absl import flags
import tensorflow as tf
from mediapipe.util.sequence import media_sequence as ms

FLAGS = flags.FLAGS
SECONDS_TO_MICROSECONDS = 1000000


def bytes23(string):
  """Creates a bytes string in either Python 2 or  3."""
  if sys.version_info >= (3, 0):
    return bytes(string, 'utf8')
  else:
    return bytes(string)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not flags.FLAGS.path_to_input_video:
    raise ValueError('You must specify the path to the input video.')
  metadata = tf.train.SequenceExample()
  ms.set_clip_data_path(bytes23(flags.FLAGS.path_to_input_video), metadata)
  ms.set_clip_start_timestamp(0, metadata)
  ms.set_clip_end_timestamp(
      int(float(300 * SECONDS_TO_MICROSECONDS)), metadata)
  with open('/tmp/mediapipe/metadata.tfrecord', 'wb') as writer:
    writer.write(metadata.SerializeToString())


if __name__ == '__main__':
  flags.DEFINE_string('path_to_input_video', '', 'Path to the input video.')
  app.run(main)
