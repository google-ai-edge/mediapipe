## Face Detection on Coral with Webcam

MediaPipe is able to run cross platform across device types like desktop, mobile
and edge devices. Here is an example of running MediaPipe
[face detection pipeline](./face_detection_desktop.md) on edge device like
[Google Coral dev board](https://coral.withgoogle.com/products/dev-board) with
[Edge TPU](https://cloud.google.com/edge-tpu/). This MediaPipe Coral face
detection pipeline is running [coral specific quantized version](https://github.com/google/mediapipe/blob/master/mediapipe/examples/coral/models/face-detector-quantized_edgetpu.tflite)
of the [MediaPipe face detection TFLite model](https://github.com/google/mediapipe/blob/master/mediapipe/models/face_detection_front.tflite)
accelerated on Edge TPU.

### Cross compilation of MediaPipe Coral binaries in Docker

We recommend building the MediaPipe binaries not on the edge device due to
limited compute resulting in long build times. Instead, we will build MediaPipe
binaries using Docker containers on a more powerful host machine. For step by
step details of cross compiling and running MediaPipe binaries on Coral dev
board, please refer to [README.md in MediaPipe Coral example folder](https://github.com/google/mediapipe/blob/master/mediapipe/examples/coral/README.md).

![Face Detection running on Coral](images/face_detection_demo_coral.jpg)
