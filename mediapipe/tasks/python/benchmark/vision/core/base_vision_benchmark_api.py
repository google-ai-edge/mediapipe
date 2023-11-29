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
import time

VISION_TEST_DATA_DIR = 'mediapipe/tasks/testdata/vision'


def benchmark_task(func, image, n_iterations):
  """Collect inference times for a given task after benchmarking.

  Args:
      func: The task function used for benchmarking.
      image: The input MediaPipe Image.
      n_iterations: Number of iterations to run the benchmark.

  Returns:
      List of inference times in milliseconds.
  """
  inference_times = []

  for _ in range(n_iterations):
    start_time_ns = time.time_ns()
    # Run the method for the task (e.g., classify)
    func(image)
    end_time_ns = time.time_ns()
    inference_times.append((end_time_ns - start_time_ns) / 1_000_000)

  return inference_times
