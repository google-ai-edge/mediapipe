# MediaPipe Image Classifier Benchmark

## Download the repository

First, clone this Git repo.

Run this script to install the required dependencies and download the TFLite models:

```
cd mediapipe/mediapipe/tasks/python/benchmark/vision/image_classifier
sh setup.sh
```

## Run the benchmark
```
python3 image_classifier_benchmark.py
```
*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `classifier.tflite`
    *   TensorFlow Lite image classification models **with metadata**  
        * Models from [TensorFlow Hub](https://tfhub.dev/tensorflow/collections/lite/task-library/image-classifier/1)
        * Models from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/image_classifier/index#models)
        * Models trained with [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/customization/image_classifier) are supported.
*   You can optionally specify the `iterations` parameter to limit the number of
    iterations for benchmarking:
    *   Supported value: A positive integer.
    *   Default value: `100`
*   Example usage:
    ```
    python3 image_classifier_benchmark.py \
      --model classifier.tflite \
      --iterations 200
    ```
