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
"""Demo for making a text classifier model by MediaPipe Model Maker."""

import os
import tempfile

# Dependency imports

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from mediapipe.model_maker.python.core.utils import quantization
from mediapipe.model_maker.python.text import text_classifier

FLAGS = flags.FLAGS


def define_flags():
  flags.DEFINE_string('export_dir', None,
                      'The directory to save exported files.')
  flags.DEFINE_enum('supported_model', 'average_word_embedding',
                    ['average_word_embedding', 'bert'],
                    'The text classifier to run.')
  flags.mark_flag_as_required('export_dir')


def download_demo_data():
  """Downloads demo data, and returns directory path."""
  data_path = tf.keras.utils.get_file(
      fname='SST-2.zip',
      origin='https://dl.fbaipublicfiles.com/glue/data/SST-2.zip',
      extract=True)
  return os.path.join(os.path.dirname(data_path), 'SST-2')  # folder name


def run(data_dir,
        export_dir=tempfile.mkdtemp(),
        supported_model=(
            text_classifier.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER)):
  """Runs demo."""

  # Gets training data and validation data.
  csv_params = text_classifier.CSVParams(
      text_column='sentence', label_column='label', delimiter='\t')
  train_data = text_classifier.Dataset.from_csv(
      filename=os.path.join(os.path.join(data_dir, 'train.tsv')),
      csv_params=csv_params)
  validation_data = text_classifier.Dataset.from_csv(
      filename=os.path.join(os.path.join(data_dir, 'dev.tsv')),
      csv_params=csv_params)

  quantization_config = None
  if (supported_model ==
      text_classifier.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER):
    hparams = text_classifier.AverageWordEmbeddingHParams(
        epochs=10, batch_size=32, learning_rate=0, export_dir=export_dir
    )
  # Warning: This takes extremely long to run on CPU
  elif (
      supported_model == text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER):
    quantization_config = quantization.QuantizationConfig.for_dynamic()
    hparams = text_classifier.BertHParams(
        epochs=3, batch_size=48, learning_rate=3e-5, export_dir=export_dir
    )

  # Fine-tunes the model.
  options = text_classifier.TextClassifierOptions(
      supported_model=supported_model, hparams=hparams)
  model = text_classifier.TextClassifier.create(train_data, validation_data,
                                                options)

  # Gets evaluation results.
  metrics = model.evaluate(validation_data)
  print('Eval accuracy: %f' % metrics[1])

  model.export_model(quantization_config=quantization_config)
  model.export_labels(export_dir=options.hparams.export_dir)


def main(_):
  logging.set_verbosity(logging.INFO)
  data_dir = download_demo_data()
  export_dir = os.path.expanduser(FLAGS.export_dir)

  if FLAGS.supported_model == 'average_word_embedding':
    supported_model = (
        text_classifier.SupportedModels.AVERAGE_WORD_EMBEDDING_CLASSIFIER)
  elif FLAGS.supported_model == 'bert':
    supported_model = text_classifier.SupportedModels.MOBILEBERT_CLASSIFIER

  run(data_dir, export_dir, supported_model)


if __name__ == '__main__':
  define_flags()
  app.run(main)
