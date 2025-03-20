# Copyright 2022 The MediaPipe Authors.
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
"""Demo for making an gesture recognizer model by Mediapipe Model Maker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

from mediapipe.model_maker.python.vision import gesture_recognizer

FLAGS = flags.FLAGS

# TODO: Move hand gesture recognizer demo dataset to an
# open-sourced directory.
TEST_DATA_DIR = 'mediapipe/model_maker/python/vision/gesture_recognizer/testdata/raw_data'


def define_flags():
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string('input_data_dir', None,
                      'The directory with input training data.')
  flags.mark_flag_as_required('export_dir')


def run(data_dir: str, export_dir: str):
  """Runs demo."""
  data = gesture_recognizer.Dataset.from_folder(dirname=data_dir)
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)

  model = gesture_recognizer.GestureRecognizer.create(
      train_data=train_data,
      validation_data=validation_data,
      options=gesture_recognizer.GestureRecognizerOptions(
          hparams=gesture_recognizer.HParams(export_dir=export_dir)))

  metric = model.evaluate(test_data, batch_size=2)
  print('Evaluation metric')
  print(metric)

  model.export_model()


def main(_):
  logging.set_verbosity(logging.INFO)

  if FLAGS.input_data_dir is None:
    data_dir = os.path.join(FLAGS.test_srcdir, TEST_DATA_DIR)
  else:
    data_dir = FLAGS.input_data_dir

  export_dir = os.path.expanduser(FLAGS.export_dir)
  run(data_dir=data_dir, export_dir=export_dir)


if __name__ == '__main__':
  define_flags()
  app.run(main)
