# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
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
"""CLI tool for display metadata."""

from absl import app
from absl import flags

from mediapipe.tasks.python.metadata import metadata

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None, 'Path to the TFLite model file.')
flags.DEFINE_string('export_json_path', None, 'Path to the output JSON file.')


def main(_):
  displayer = metadata.MetadataDisplayer.with_model_file(FLAGS.model_path)
  with open(FLAGS.export_json_path, 'w') as f:
    f.write(displayer.get_metadata_json())


if __name__ == '__main__':
  app.run(main)
