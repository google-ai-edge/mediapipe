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
"""Benchmark for the image classifier task."""
import argparse
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_IMAGE_FILE = 'burger.jpg'


def run(model: str, n_iterations: int, delegate: python.BaseOptions.Delegate):
    """Run asynchronous inference on images and benchmark.

    Args:
        model: Path to the TFLite model.
        n_iterations: Number of iterations to run the benchmark.
        delegate: CPU or GPU delegate for inference.
    """
    inference_times = []
    
    # Initialize the image classifier
    base_options = python.BaseOptions(model_asset_path=model, delegate=delegate)
    options = vision.ImageClassifierOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE,
        max_results=1)
    classifier = vision.ImageClassifier.create_from_options(options)
    mp_image = mp.Image.create_from_file(_IMAGE_FILE)

    for _ in range(n_iterations):
        start_time_ns = time.time_ns()
        classifier.classify(mp_image)
        end_time_ns = time.time_ns()
        # Convert to milliseconds
        inference_times.append((end_time_ns - start_time_ns) / 1_000_000)

    classifier.close()
    return np.percentile(inference_times, 95)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='Path to image classification model.', required=True)
    parser.add_argument(
        '--iterations', help='Number of iterations for benchmarking.', type=int,
        default=100)
    args = parser.parse_args()

    # Run benchmark on CPU
    cpu_time = run(args.model, args.iterations, python.BaseOptions.Delegate.CPU)
    print(f"95th Percentile Inference Time on CPU: {cpu_time:.6f} milliseconds")

    # Run benchmark on GPU
    gpu_time = run(args.model, args.iterations, python.BaseOptions.Delegate.GPU)
    print(f"95th Percentile Inference Time on GPU: {gpu_time:.6f} milliseconds")


if __name__ == '__main__':
    main()
