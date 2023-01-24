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
"""Utilities for files."""

import os

# resources dependency


def get_absolute_path(file_path: str) -> str:
  """Gets the absolute path of a file in the model_maker directory.

  Args:
    file_path: The path to a file relative to the `mediapipe` dir

  Returns:
   The full path of the file
  """
  # Extract the file path before and including 'model_maker' as the
  # `mm_base_dir`. By joining it with the `path` after 'model_maker/', it
  # yields to the absolute path of the model files directory. We must join
  # on 'model_maker' because in the pypi package, the 'model_maker' directory
  # is renamed to 'mediapipe_model_maker'. So we have to join on model_maker
  # to ensure that the `mm_base_dir` path includes the renamed
  # 'mediapipe_model_maker' directory.
  cwd = os.path.dirname(__file__)
  cwd_stop_idx = cwd.rfind('model_maker') + len('model_maker')
  mm_base_dir = cwd[:cwd_stop_idx]
  file_path_start_idx = file_path.find('model_maker') + len('model_maker') + 1
  mm_relative_path = file_path[file_path_start_idx:]
  absolute_path = os.path.join(mm_base_dir, mm_relative_path)
  return absolute_path
