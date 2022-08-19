# Image Style

This graph performs face aligning.

## Getting started

Clone branch.

1. Desktop-CPU

Build with:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/image_style:image_style_cpu
```
Run with (using your camera):
```
bazel-bin/mediapipe/examples/desktop/image_style/image_style_cpu --calculator_graph_config_file=mediapipe/graphs/image_style/image_style_cpu.pbtxt
```
Run with (using video):
```
bazel-bin/mediapipe/examples/desktop/image_style/image_style_cpu --calculator_graph_config_file=mediapipe/graphs/image_style/image_style_cpu.pbtxt --input_video_path=/path/video.mp4 --output_video_path=/path/outvideo.mp4
```

2. Mobile (Android)

Build with:
```
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/imagestylegpu:imagestylegpu
```
Install with:
```
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/imagestylegpu/imagestylegpu.apk
```

