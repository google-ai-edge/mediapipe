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
"""MediaPipe image segmenter benchmark."""

import argparse

from mediapipe.python._framework_bindings import image
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import image_segmenter
from mediapipe.tasks.python.benchmark import benchmark_utils
from mediapipe.tasks.python.benchmark.vision.core import base_vision_benchmark_api

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'segmentation_input_rotation0.jpg'


def run(
    model: str,
    n_iterations: int,
    delegate: base_options.BaseOptions.Delegate,
    percentile: float,
):
  """Run an image segmentation benchmark.

  Args:
      model: Path to the TFLite model.
      n_iterations: Number of iterations to run the benchmark.
      delegate: CPU or GPU delegate for inference.
      percentile: Percentage for the percentiles to compute. Values must be
        between 0 and 100 inclusive.

  Returns:
    The n-th percentile of the inference times.
  """
  # Initialize the image segmenter
  default_model_path = benchmark_utils.get_test_data_path(
      base_vision_benchmark_api.VISION_TEST_DATA_DIR, _MODEL_FILE
  )
  model_path = benchmark_utils.get_model_path(model, default_model_path)
  options = image_segmenter.ImageSegmenterOptions(
      base_options=base_options.BaseOptions(
          model_asset_path=model_path, delegate=delegate
      ),
      output_confidence_masks=True, output_category_mask=True
  )

  with image_segmenter.ImageSegmenter.create_from_options(options) as segmenter:
    mp_image = image.Image.create_from_file(
        benchmark_utils.get_test_data_path(
            base_vision_benchmark_api.VISION_TEST_DATA_DIR, _IMAGE_FILE
        )
    )
    # Run the benchmark and return the nth percentile of the inference times
    nth_percentile = base_vision_benchmark_api.nth_percentile(
        segmenter.segment, mp_image, n_iterations, percentile
    )
  return nth_percentile


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
      '--model',
      help='Path to image segmentation model.',
      required=False,
      default=None,
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
