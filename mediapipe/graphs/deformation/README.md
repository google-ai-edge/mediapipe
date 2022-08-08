# Deformation

this graph performs face processing

## Getting started

Clone branch.

1. Deformation-CPU

Build with:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/deformation:deformation_cpu
```
Run with (using your camera):
```
bazel-bin/mediapipe/examples/desktop/deformation/deformation_cpu 
--calculator_graph_config_file=mediapipe/graphs/deformation/deformation_cpu.pbtxt
```
Run with (using video):
```
bazel-bin/mediapipe/examples/desktop/deformation/deformation_cpu --calculator_graph_config_file=mediapipe/graphs/deformation/deformation_cpu.pbtxt --input_video_path=/path/video.mp4 --output_video_path=/path/outvideo.mp4
```

2. Mobile (Android)

Build with:
```
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/deformation:deformationgpu
```
Install with:
```
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/deformation/deformationgpu.apk
```
