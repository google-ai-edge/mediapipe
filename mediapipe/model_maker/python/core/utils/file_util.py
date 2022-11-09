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
  """Gets the absolute path of a file.

  Args:
    file_path: The path to a file relative to the `mediapipe` dir

  Returns:
   The full path of the file
  """
  # Extract the file path before mediapipe/ as the `base_dir`. By joining it
  # with the `path` which defines the relative path under mediapipe/, it
  # yields to the absolute path of the model files directory.
  cwd = os.path.dirname(__file__)
  base_dir = cwd[:cwd.rfind('mediapipe')]
  absolute_path = os.path.join(base_dir, file_path)
  return absolute_path
