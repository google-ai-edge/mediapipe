This directory contains example MediaPipe applications on iOS.

|  Use Case                             |  Directory                          |
|---------------------------------------|:-----------------------------------:|
|  Edge Detection on GPU                |  edgedetection                   |
|  Face Detection on CPU                |  facedetectioncpu                   |
|  Face Detection on GPU                |  facedetectiongpu                   |
|  Object Detection on CPU              |  objectdetectioncpu                 |
|  Object Detection on GPU              |  objectdetectiongpu                 |
|  Hand Detection on GPU                |  handdetectiongpu                   |
|  Hand Tracking on GPU                 |  handtrackinggpu                    |

For instance, to build an example app for face detection on CPU, run:

```bash
bazel build -c opt --config=ios_arm64 --xcode_version=$XCODE_VERSION --cxxopt='-std=c++14' mediapipe/examples/ios/facedetectioncpu:FaceDetectionCpuApp
```
(Note: with your own $XCODE_VERSION)
