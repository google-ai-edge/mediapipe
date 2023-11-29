# MediaPipe Image Classifier Benchmark

## Download the repository

First, clone this Git repo.

Run this commands to download the TFLite models and image files:

```
cd mediapipe/mediapipe/tasks/python/benchmark/vision/image_classifier
wget -O classifier.tflite -q https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite
```

## Run the benchmark
```
bazel run -c opt //mediapipe/tasks/python/benchmark/vision/image_classifier:image_classifier_benchmark
```
*   You can optionally specify the `model` parameter to set the TensorFlow Lite
    model to be used:
    *   The default value is `mobilenet_v2_1.0_224.tflite`
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
    bazel run -c opt :image_classifier_benchmark -- \
      --model classifier.tflite \
      --iterations 200
    ```
