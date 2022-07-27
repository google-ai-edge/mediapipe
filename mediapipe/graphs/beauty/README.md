# Beauty

this graph performs face processing

## Getting started

Clone branch.

1. Desktop-CPU (Divided calculators)

Build with:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/beauty:beauty_cpu
```
Run with (using your camera):
```
bazel-bin/mediapipe/examples/desktop/beauty/beauty_cpu 
--calculator_graph_config_file=mediapipe/graphs/beauty/beauty_desktop_cpu.pbtxt
```
Run with (using video):
```
bazel-bin/mediapipe/examples/desktop/beauty/beauty_cpu 
--calculator_graph_config_file=mediapipe/graphs/beauty/beauty_desktop_cpu.pbtxt 
--input_video_path=/path/video.mp4 
--output_video_path=/path/outvideo.mp4
```

2. Desktop-CPU-Single (Not divided, using render data)

Build with:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/beauty:beauty_cpu_single
```
Run with (using your camera):
```
bazel-bin/mediapipe/examples/desktop/beauty/beauty_cpu_single 
--calculator_graph_config_file=mediapipe/graphs/beauty/beauty_desktop_cpu_single.pbtxt
```
Run with (using video):
```
bazel-bin/mediapipe/examples/desktop/beauty/beauty_cpu_single 
--calculator_graph_config_file=mediapipe/graphs/beauty/beauty_desktop_cpu_single.pbtxt 
--input_video_path=/path/video.mp4 
--output_video_path=/path/outvideo.mp4
```
3. Mobile (Android)

Build with:
```
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/beauty:beautygpu
```
Install with:
```
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/beauty/beautygpu.apk
```

4. Mobile-Single (Android)

Build with:
```
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/beauty_single:beautygpusingle
```
Install with:
```
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/beauty_single/beautygpusingle.apk
```
