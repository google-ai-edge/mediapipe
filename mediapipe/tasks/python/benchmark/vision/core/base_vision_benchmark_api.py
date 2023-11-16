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
"""MediaPipe vision benchmark base api."""
import os
import time
import numpy as np

VISION_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def nth_percentile(func, image, n_iterations, percentile):
  """Run a nth percentile benchmark for a given task using the function.

  Args:
      func: The method associated with a given task used for benchmarking.
      image: The input MediaPipe Image.
      n_iterations: Number of iterations to run the benchmark.
      percentile: Percentage for the percentiles to compute. Values must be
        between 0 and 100 inclusive.

  Returns:
    The n-th percentile of the inference times in milliseconds.
  """
  inference_times = []

  for _ in range(n_iterations):
    start_time_ns = time.time_ns()
    # Run the method for the task (e.g., classify)
    func(image)
    end_time_ns = time.time_ns()
    inference_times.append((end_time_ns - start_time_ns) / 1_000_000)

  return np.percentile(inference_times, percentile)
