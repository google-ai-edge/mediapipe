import argparse
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_IMAGE_FILE = 'burger.jpg'


def run(model: str, n_iterations: int, delegate: python.BaseOptions.Delegate,
        percentile: float):
    """Run asynchronous inference on images and benchmark.

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
    return np.percentile(inference_times, percentile)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='Path to image classification model.', required=True,
        default='classifier.tflite')
    parser.add_argument(
        '--iterations', help='Number of iterations for benchmarking.', type=int,
        default=100)
    parser.add_argument(
        '--percentile', help='Percentile for benchmarking statistics.',
        type=float, default=95.0)
    args = parser.parse_args()

    # Run benchmark on CPU
    cpu_time = run(args.model, args.iterations, python.BaseOptions.Delegate.CPU,
                   args.percentile)
    print(f"{args.percentile}th Percentile Inference Time on CPU: "
          f"{cpu_time:.6f} milliseconds")

    # Run benchmark on GPU
    gpu_time = run(args.model, args.iterations, python.BaseOptions.Delegate.GPU,
                   args.percentile)
    print(f"{args.percentile}th Percentile Inference Time on GPU: "
          f"{gpu_time:.6f} milliseconds")


if __name__ == '__main__':
    main()
