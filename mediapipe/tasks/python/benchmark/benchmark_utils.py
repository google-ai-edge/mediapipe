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
"""Benchmark utils for MediaPipe Tasks."""

import os
import numpy as np


def nth_percentile(inference_times, percentile):
  """Calculate the nth percentile of the inference times."""
  return np.percentile(inference_times, percentile)


def average(inference_times):
  """Calculate the average of the inference times."""
  return np.mean(inference_times)


def get_test_data_path(test_srcdir, file_or_dirname_path: str) -> str:
  """Determine the test data path.

  Args:
      test_srcdir: The path to the test source directory.
      file_or_dirname_path: The path to the file or directory.

  Returns:
      The full test data path.
  """
  for directory, subdirs, files in os.walk(test_srcdir):
    for f in subdirs + files:
      path = os.path.join(directory, f)
      if path.endswith(file_or_dirname_path):
        return path
  raise ValueError(
      "No %s in test directory: %s." % (file_or_dirname_path, test_srcdir)
  )


def get_model_path(custom_model, default_model_path):
  """Determine the model path based on the existence of the custom model.

  Args:
      custom_model: The path to the custom model provided by the user.
      default_model_path: The path to the default model.

  Returns:
      The path to the model to be used.
  """
  if custom_model is not None and os.path.exists(custom_model):
    print(f"Using provided model: {custom_model}")
    return custom_model
  else:
    if custom_model is not None:
      print(
          f"Warning: Provided model '{custom_model}' not found. "
          "Using default model instead."
      )
    print(f"Using default model: {default_model_path}")
    return default_model_path
