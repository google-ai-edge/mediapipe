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
"""Demo for making an image classifier model by MediaPipe Model Maker."""

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.vision import image_classifier

FLAGS = flags.FLAGS


def define_flags() -> None:
  """Define flags for the image classifier model maker demo."""
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_string(
      'input_data_dir', None,
      """The directory with input training data. If the training data is not
      specified, the pipeline will download a default training dataset.""")
  flags.DEFINE_enum_class('spec',
                          image_classifier.SupportedModels.EFFICIENTNET_LITE0,
                          image_classifier.SupportedModels,
                          'The image classifier to run.')
  flags.DEFINE_enum('quantization', None, ['dynamic', 'int8', 'float16'],
                    'The quantization method to use when exporting the model.')
  flags.mark_flag_as_required('export_dir')


def download_demo_data() -> str:
  """Downloads demo data, and returns directory path."""
  data_dir = tf.keras.utils.get_file(
      fname='flower_photos.tgz',
      origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      extract=True)
  return os.path.join(os.path.dirname(data_dir), 'flower_photos')  # folder name


def run(data_dir: str, export_dir: str,
        model_spec: image_classifier.SupportedModels,
        quantization_option: str) -> None:
  """Runs demo."""
  data = image_classifier.Dataset.from_folder(data_dir)
  train_data, rest_data = data.split(0.8)
  validation_data, test_data = rest_data.split(0.5)
  model_options = image_classifier.ImageClassifierOptions(
      supported_model=model_spec,
      hparams=image_classifier.HParams(export_dir=export_dir),
  )
  model = image_classifier.ImageClassifier.create(
      train_data=train_data,
      validation_data=validation_data,
      options=model_options)

  _, acc = model.evaluate(test_data)
  print('Test accuracy: %f' % acc)

  if quantization_option is None:
    quantization_config = None
  elif quantization_option == 'dynamic':
    quantization_config = quantization.QuantizationConfig.for_dynamic()
  elif quantization_option == 'int8':
    quantization_config = quantization.QuantizationConfig.for_int8(train_data)
  elif quantization_option == 'float16':
    quantization_config = quantization.QuantizationConfig.for_float16()
  else:
    raise ValueError(f'Quantization: {quantization} is not recognized')

  model.export_model(quantization_config=quantization_config)


def main(_) -> None:
  logging.set_verbosity(logging.INFO)

  if FLAGS.input_data_dir is None:
    data_dir = download_demo_data()
  else:
    data_dir = FLAGS.input_data_dir

  export_dir = os.path.expanduser(FLAGS.export_dir)
  run(data_dir=data_dir,
      export_dir=export_dir,
      model_spec=FLAGS.spec,
      quantization_option=FLAGS.quantization)


if __name__ == '__main__':
  define_flags()
  app.run(main)
