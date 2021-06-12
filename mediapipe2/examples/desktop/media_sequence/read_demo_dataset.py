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

# Lint as: python3
"""Example of reading a MediaSequence dataset.
"""

from absl import app
from absl import flags

from mediapipe.examples.desktop.media_sequence.demo_dataset import DemoDataset
import tensorflow as tf

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  demo_data_path = '/tmp/demo_data/'
  with tf.Graph().as_default():
    d = DemoDataset(demo_data_path)
    dataset = d.as_dataset('test')
    # implement additional processing and batching here
    dataset_output = dataset.make_one_shot_iterator().get_next()
    images = dataset_output['images']
    labels = dataset_output['labels']

    with tf.Session() as sess:
      images_, labels_ = sess.run([images, labels])
    print('The shape of images_ is %s' % str(images_.shape))  # pylint: disable=superfluous-parens
    print('The shape of labels_ is %s' % str(labels_.shape))  # pylint: disable=superfluous-parens


if __name__ == '__main__':
  app.run(main)
