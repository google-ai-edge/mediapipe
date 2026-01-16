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
"""Generate Java reference docs for MediaPipe."""
import pathlib
import shutil

from absl import app
from absl import flags

from tensorflow_docs.api_generator import gen_java

_OUT_DIR = flags.DEFINE_string('output_dir', '/tmp/mp_java/',
                               'Write docs here.')

_SITE_PATH = flags.DEFINE_string('site_path', '/mediapipe/api_docs/java',
                                 'Path prefix in the _toc.yaml')

_ = flags.DEFINE_string('code_url_prefix', None,
                        '[UNUSED] The url prefix for links to code.')

_ = flags.DEFINE_bool(
    'search_hints', True,
    '[UNUSED] Include metadata search hints in the generated files')

_ANDROID_SDK = pathlib.Path('android/sdk/api/26.txt')


def main(_) -> None:
  # Default to using a relative path to find the Java source.
  mp_root = pathlib.Path(__file__)
  while (mp_root := mp_root.parent).name != 'mediapipe':
    # Find the nearest `mediapipe` dir.
    if not mp_root.name:
      # We've hit the filesystem root - abort.
      raise FileNotFoundError('"mediapipe" root not found')

  # Find the root from which all packages are relative.
  root = mp_root.parent

  # Externally, parts of the repo are nested inside a mediapipe/ directory
  # that does not exist internally. Support both.
  if (mp_root / 'mediapipe').exists():
    mp_root = mp_root / 'mediapipe'

  # We need to copy this into the tasks dir to ensure we don't leave broken
  # links in the generated docs.
  old_api_dir = 'java/com/google/mediapipe/framework/image'
  shutil.copytree(
      mp_root / old_api_dir,
      mp_root / 'tasks' / old_api_dir,
      dirs_exist_ok=True)

  gen_java.gen_java_docs(
      package='com.google.mediapipe',
      source_path=mp_root / 'tasks/java',
      output_dir=pathlib.Path(_OUT_DIR.value),
      site_path=pathlib.Path(_SITE_PATH.value),
      federated_docs={'https://developer.android.com': root / _ANDROID_SDK})


if __name__ == '__main__':
  app.run(main)
