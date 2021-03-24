---
layout: default
title: MediaPipe in C++
parent: Getting Started
has_children: true
has_toc: false
nav_order: 5
---

# MediaPipe in C++
{: .no_toc }

1. TOC
{:toc}
---

Please follow instructions below to build C++ command-line example apps in the
supported MediaPipe [solutions](../solutions/solutions.md). To learn more about
these example apps, start from [Hello World! in C++](./hello_world_cpp.md).

## Building C++ command-line example apps

### Option 1: Running on CPU

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible.

### Option 2: Running on GPU

Note: This currently works only on Linux, and please first follow
[OpenGL ES Setup on Linux Desktop](./gpu_support.md#opengl-es-setup-on-linux-desktop).

1.  To build, for example, [MediaPipe Hands](../solutions/hands.md), run:

    ```bash
    bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
      mediapipe/examples/desktop/hand_tracking:hand_tracking_gpu
    ```

2.  To run the application:

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_gpu \
      --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_mobile.pbtxt
    ```

    This will open up your webcam as long as it is connected and on. Any errors
    is likely due to your webcam being not accessible, or GPU drivers not setup
    properly.
