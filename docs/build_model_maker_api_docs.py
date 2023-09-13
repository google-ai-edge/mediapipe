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
# ==============================================================================
r"""MediaPipe Model Maker reference docs generation script.

This script generates API reference docs for the `mediapipe` PIP package.

$> pip install -U git+https://github.com/tensorflow/docs mediapipe-model-maker
$> python build_model_maker_api_docs.py
"""

import os

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib

try:
  # mediapipe has not been set up to work with bazel yet, so catch & report.
  import mediapipe_model_maker  # pytype: disable=import-error
except ImportError as e:
  raise ImportError('Please `pip install mediapipe-model-maker`.') from e


PROJECT_SHORT_NAME = 'mediapipe_model_maker'
PROJECT_FULL_NAME = 'MediaPipe Model Maker'

_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    default='/tmp/generated_docs',
    help='Where to write the resulting docs.')

_URL_PREFIX = flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/google/mediapipe/tree/master/mediapipe/model_maker',
    'The url prefix for links to code.')

_SEARCH_HINTS = flags.DEFINE_bool(
    'search_hints', True,
    'Include metadata search hints in the generated files')

_SITE_PATH = flags.DEFINE_string('site_path', '/mediapipe/api_docs/python',
                                 'Path prefix in the _toc.yaml')


def gen_api_docs():
  """Generates API docs for the mediapipe-model-maker package."""

  doc_generator = generate_lib.DocGenerator(
      root_title=PROJECT_FULL_NAME,
      py_modules=[(PROJECT_SHORT_NAME, mediapipe_model_maker)],
      base_dir=os.path.dirname(mediapipe_model_maker.__file__),
      code_url_prefix=_URL_PREFIX.value,
      search_hints=_SEARCH_HINTS.value,
      site_path=_SITE_PATH.value,
      callbacks=[],
  )

  doc_generator.build(_OUTPUT_DIR.value)

  print('Docs output to:', _OUTPUT_DIR.value)


def main(_):
  gen_api_docs()


if __name__ == '__main__':
  app.run(main)
