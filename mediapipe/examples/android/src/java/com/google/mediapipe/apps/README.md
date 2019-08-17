MediaPipe Examples
==================

This directory contains MediaPipe Android example applications for different use cases. The applications use CameraX API to access the camera.

## Use Cases

|  Use Case                             |  Directory                          |
|---------------------------------------|:-----------------------------------:|
|  Edge Detection on GPU                |  edgedetectiongpu                   |
|  Face Detection on CPU                |  facedetectioncpu                   |
|  Face Detection on GPU                |  facedetectiongpu                   |
|  Object Detection on CPU              |  objectdetectioncpu                 |
|  Object Detection on GPU              |  objectdetectiongpu                 |
|  Hair Segmentation on GPU             |  hairsegmentationgpu                |
|  Hand Detection on GPU                |  handdetectiongpu                   |
|  Hand Tracking on GPU                 |  handtrackinggpu                    |

For instance, to build an example app for face detection on CPU, run:

```bash
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu
```

To further install the app on an Android device, run:

```bash
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu/facedetectioncpu.apk
```
