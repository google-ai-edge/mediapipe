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
"""MediaPipe vision benchmarker."""

import argparse

from mediapipe.tasks.python.benchmark import benchmark_utils as bu
from mediapipe.tasks.python.benchmark.vision.core import base_vision_benchmark_api
from mediapipe.tasks.python.core import base_options


def benchmarker(benchmark_function, default_model_name):
  """Executes a benchmarking process using a specified function ann model.

  Args:
      benchmark_function: A callable function to be executed for benchmarking.
        This function should contain the logic of the task to be benchmarked and
        should be capable of utilizing a model specified by its name.
      default_model_name: The name or path of the default model to be used in
        the benchmarking process. This is useful when the benchmarking function
        requires a model and no other model is explicitly specified.
  """
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
      '--mode',
      help='Benchmarking mode (e.g., "nth_percentile").',
      required=False,
      default='nth_percentile',
  )
  parser.add_argument('--model', help='Path to the model.', default=None)
  parser.add_argument(
      '--iterations',
      help='Number of iterations for benchmarking.',
      type=int,
      default=100,
  )
  parser.add_argument(
      '--percentile',
      help='Percentile for benchmarking statistics.',
      type=float,
      default=95.0,
  )

  args = parser.parse_args()

  # Get the model path
  default_model_path = bu.get_test_data_path(
      base_vision_benchmark_api.VISION_TEST_DATA_DIR, default_model_name
  )
  model_path = bu.get_model_path(args.model, default_model_path)

  # Define a mapping of modes to their respective function argument lists
  mode_args_mapping = {
      'nth_percentile': {'percentile': args.percentile},
      'average': {},
  }

  # Check if the mode is supported and get the argument dictionary
  if args.mode not in mode_args_mapping:
    raise ValueError(f'Unsupported benchmarking mode: {args.mode}')

  mode_args = mode_args_mapping[args.mode]

  # Run the benchmark for both CPU and GPU and calculate results based on mode
  results = {}
  for delegate_type in [
      base_options.BaseOptions.Delegate.CPU,
      base_options.BaseOptions.Delegate.GPU,
  ]:
    inference_times = benchmark_function(
        model_path, args.iterations, delegate_type
    )

    # Calculate the benchmark result based on the mode
    if args.mode == 'nth_percentile':
      results[delegate_type] = bu.nth_percentile(inference_times, **mode_args)
    elif args.mode == 'average':
      results[delegate_type] = bu.average(inference_times)

  # Report benchmarking results
  for delegate_type, result in results.items():
    print(
        f'Inference time {delegate_type} {mode_args_mapping[args.mode]}: '
        f'{result:.6f} milliseconds'
    )
