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
"""Demo for making an object detector model by MediaPipe Model Maker."""

import os
# Dependency imports

from absl import app
from absl import flags
from absl import logging

from mediapipe.model_maker.python.vision import object_detector

FLAGS = flags.FLAGS

TEST_DATA_DIR = 'mediapipe/model_maker/python/vision/object_detector/testdata/coco_data'


def define_flags() -> None:
  """Define flags for the object detection model maker demo."""
  flags.DEFINE_string(
      'export_dir', None, 'The directory to save exported files.'
  )
  flags.DEFINE_string(
      'input_data_dir',
      None,
      """The directory with input training data. If the training data is not
      specified, the pipeline will use the test dataset.""",
  )
  flags.DEFINE_bool('qat', True, 'Whether or not to do QAT.')
  flags.mark_flag_as_required('export_dir')


def run(data_dir: str, export_dir: str, qat: bool):
  """Runs demo."""
  data = object_detector.Dataset.from_coco_folder(data_dir)
  train_data, rest_data = data.split(0.6)
  validation_data, test_data = rest_data.split(0.5)

  hparams = object_detector.HParams(batch_size=1, export_dir=export_dir)
  options = object_detector.ObjectDetectorOptions(
      supported_model=object_detector.SupportedModels.MOBILENET_V2,
      hparams=hparams,
  )
  model = object_detector.ObjectDetector.create(
      train_data=train_data, validation_data=validation_data, options=options
  )
  loss, coco_metrics = model.evaluate(test_data, batch_size=1)
  print(f'Evaluation loss:{loss}, coco_metrics:{coco_metrics}')
  if qat:
    qat_hparams = object_detector.QATHParams(batch_size=1)
    model.quantization_aware_training(train_data, validation_data, qat_hparams)
    qat_loss, qat_coco_metrics = model.evaluate(test_data, batch_size=1)
    print(f'QAT Evaluation loss:{qat_loss}, coco_metrics:{qat_coco_metrics}')

  model.export_model()


def main(_) -> None:
  logging.set_verbosity(logging.INFO)

  if FLAGS.input_data_dir is None:
    data_dir = os.path.join(FLAGS.test_srcdir, TEST_DATA_DIR)
  else:
    data_dir = FLAGS.input_data_dir

  export_dir = os.path.expanduser(FLAGS.export_dir)
  run(data_dir=data_dir, export_dir=export_dir, qat=FLAGS.qat)


if __name__ == '__main__':
  define_flags()
  app.run(main)
