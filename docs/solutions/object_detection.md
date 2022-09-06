---
layout: default
title: Object Detection
parent: Solutions
nav_order: 9
---

# MediaPipe Object Detection
{: .no_toc }

<details close markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
---

![object_detection_android_gpu.gif](https://mediapipe.dev/images/mobile/object_detection_android_gpu.gif)

## Example Apps

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

Please first see general instructions for
[Android](../getting_started/android.md) and [iOS](../getting_started/ios.md) on
how to build MediaPipe examples.

#### GPU Pipeline

*   Graph:
    [`mediapipe/graphs/object_detection/object_detection_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_mobile_gpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1di2ywCA_acf3y5rIcJHngWHAUNsUHAGz)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu:objectdetectiongpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectiongpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/objectdetectiongpu:ObjectDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/objectdetectiongpu/BUILD)

#### CPU Pipeline

This is very similar to the [GPU pipeline](#gpu-pipeline) except that at the
beginning and the end of the pipeline it performs GPU-to-CPU and CPU-to-GPU
image transfer respectively. As a result, the rest of graph, which shares the
same configuration as the GPU pipeline, runs entirely on CPU.

*   Graph:
    [`mediapipe/graphs/object_detection/object_detection_mobile_cpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_mobile_cpu.pbtxt))
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1eRBK6V5Qd1LCRwexitR2OXgrBBXbOfZ5)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectioncpu:objectdetectioncpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/objectdetectioncpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/objectdetectioncpu:ObjectDetectionCpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/objectdetectioncpu/BUILD)

### Desktop

#### Live Camera Input

Please first see general instructions for [desktop](../getting_started/cpp.md)
on how to build MediaPipe examples.

*   Graph:
    [`mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt)
*   Target:
    [`mediapipe/examples/desktop/object_detection:object_detection_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/object_detection/BUILD)

#### Video File Input

*   With a TFLite Model

    This uses the same
    [TFLite model](https://storage.googleapis.com/mediapipe-assets/ssdlite_object_detection.tflite)
    (see also
    [model info](https://github.com/google/mediapipe/tree/master/mediapipe/models/object_detection_saved_model/README.md))
    as in [Live Camera Input](#live-camera-input) above. The pipeline is
    implemented in this
    [graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt),
    which differs from the live-camera-input CPU-based pipeline
    [graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_mobile_cpu.pbtxt)
    simply by the additional `OpenCvVideoDecoderCalculator` and
    `OpenCvVideoEncoderCalculator` at the beginning and the end of the graph
    respectively.

    To build the application, run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_tflite
    ```

    To run the application, replace `<input video path>` and `<output video
    path>` in the command below with your own paths:

    Tip: You can find a test video available in
    `mediapipe/examples/desktop/object_detection`.

    ```
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
      --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt \
      --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
    ```

*   With a TensorFlow Model

    This uses the
    [TensorFlow model](https://github.com/google/mediapipe/tree/master/mediapipe/models/object_detection_saved_model)
    ( see also
    [model info](https://github.com/google/mediapipe/tree/master/mediapipe/models/object_detection_saved_model/README.md)),
    and the pipeline is implemented in this
    [graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_detection/object_detection_mobile_cpu.pbtxt).

    Note: The following runs TensorFlow inference on CPU. If you would like to
    run inference on GPU (Linux only), please follow
    [TensorFlow CUDA Support and Setup on Linux Desktop](../getting_started/gpu_support.md#tensorflow-cuda-support-and-setup-on-linux-desktop)
    instead.

    To build the TensorFlow CPU inference example on desktop, run:

    Note: This command also builds TensorFlow targets from scratch, and it may
    take a long time (e.g., up to 30 mins) for the first time.

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true --linkopt=-s \
    mediapipe/examples/desktop/object_detection:object_detection_tensorflow
    ```

    To run the application, replace `<input video path>` and `<output video
    path>` in the command below with your own paths:

    Tip: You can find a test video available in
    `mediapipe/examples/desktop/object_detection`.

    ```bash
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
      --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_graph.pbtxt \
      --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
    ```

### Coral

Please refer to
[these instructions](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/README.md)
to cross-compile and run MediaPipe examples on the
[Coral Dev Board](https://coral.ai/products/dev-board).

## Resources

*   [Models and model cards](./models.md#object_detection)
