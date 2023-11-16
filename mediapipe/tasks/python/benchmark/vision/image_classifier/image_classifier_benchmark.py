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
"""MediaPipe image classsifier benchmark."""

import argparse
import time
import numpy as np
from mediapipe.python._framework_bindings import image
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import image_classifier

_IMAGE_FILE = 'burger.jpg'


def run(
    model: str,
    n_iterations: int,
    delegate: base_options.BaseOptions.Delegate,
    percentile: float,
):
  """Run an image classification benchmark.

  Args:
      model: Path to the TFLite model.
      n_iterations: Number of iterations to run the benchmark.
      delegate: CPU or GPU delegate for inference.
      percentile: Percentage for the percentiles to compute. Values must be
        between 0 and 100 inclusive.

  Returns:
    The n-th percentile of the inference times.
  """
  inference_times = []

  # Initialize the image classifier
  options = image_classifier.ImageClassifierOptions(
      base_options=base_options.BaseOptions(
          model_asset_path=model, delegate=delegate
      ),
      max_results=1,
  )
  classifier = image_classifier.ImageClassifier.create_from_options(options)
  mp_image = image.Image.create_from_file(_IMAGE_FILE)

  for _ in range(n_iterations):
    start_time_ns = time.time_ns()
    classifier.classify(mp_image)
    end_time_ns = time.time_ns()
    # Convert to milliseconds
    inference_times.append((end_time_ns - start_time_ns) / 1_000_000)

  classifier.close()
  return np.percentile(inference_times, percentile)


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
      '--model',
      help='Path to image classification model.',
      required=False,
      default='classifier.tflite',
  )
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

  # Run benchmark on CPU
  cpu_time = run(
      args.model,
      args.iterations,
      base_options.BaseOptions.Delegate.CPU,
      args.percentile,
  )
  print(
      f'{args.percentile}th Percentile Inference Time on CPU: '
      f'{cpu_time:.6f} milliseconds'
  )

  # Run benchmark on GPU
  gpu_time = run(
      args.model,
      args.iterations,
      base_options.BaseOptions.Delegate.GPU,
      args.percentile,
  )
  print(
      f'{args.percentile}th Percentile Inference Time on GPU: '
      f'{gpu_time:.6f} milliseconds'
  )


if __name__ == '__main__':
  main()
