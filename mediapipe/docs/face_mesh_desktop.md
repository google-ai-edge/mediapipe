## Face Mesh on Desktop with Webcam

This doc focuses on running the **MediaPipe Face Mesh** pipeline to perform 3D
face landmark estimation in real-time on desktop with webcam input. The pipeline
internally incorporates TensorFlow Lite models. To know more about the models,
please refer to the model
[README file](https://github.com/google/mediapipe/tree/master/mediapipe/models/README.md#face-mesh).
Moreover, if you are interested in running the same pipeline on Android/iOS,
please see [Face Mesh on Android/iOS](face_mesh_mobile_gpu.md).

-   [Face Mesh on Desktop with Webcam (CPU)](#face-mesh-on-desktop-with-webcam-cpu)

-   [Face Mesh on Desktop with Webcam (GPU)](#face-mesh-on-desktop-with-webcam-gpu)

Note: Desktop GPU works only on Linux. Mesa drivers need to be installed. Please
see
[step 4 of "Installing on Debian and Ubuntu" in the installation guide](./install.md).

Note: If MediaPipe depends on OpenCV 2, please see the [known issues with OpenCV 2](#known-issues-with-opencv-2) section.

### Face Mesh on Desktop with Webcam (CPU)

To build and run Face Mesh on desktop with webcam (CPU), run:

```bash
$ bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/face_mesh:face_mesh_cpu

# It should print:
# Target //mediapipe/examples/desktop/face_mesh:face_mesh_cpu up-to-date:
#  bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu

# This will open up your webcam as long as it is connected. Errors are likely
# due to your webcam being not accessible.
$ GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu \
    --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt
```

### Face Mesh on Desktop with Webcam (GPU)

Note: please first [check that your GPU is supported](gpu.md#desktop-gpu-linux).

To build and run Face Mesh on desktop with webcam (GPU), run:

```bash
# This works only for linux currently
$ bazel build -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 \
    mediapipe/examples/desktop/face_mesh:face_mesh_gpu

# It should print:
# Target //mediapipe/examples/desktop/face_mesh:face_mesh_gpu up-to-date:
#  bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_gpu

# This will open up your webcam as long as it is connected. Errors are likely
# due to your webcam being not accessible, or GPU drivers not setup properly.
$ GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_gpu \
    --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live_gpu.pbtxt
```
