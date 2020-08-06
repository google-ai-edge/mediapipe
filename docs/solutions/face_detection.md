---
layout: default
title: Face Detection
parent: Solutions
nav_order: 1
---

# MediaPipe Face Detection
{: .no_toc }

1. TOC
{:toc}
---

## Overview

MediaPipe Face Detection is an ultrafast face detection solution that comes with
6 landmarks and multi-face support. It is based on
[BlazeFace](https://arxiv.org/abs/1907.05047), a lightweight and well-performing
face detector tailored for mobile GPU inference. The detector's super-realtime
performance enables it to be applied to any live viewfinder experience that
requires an accurate facial region of interest as an input for other
task-specific models, such as 3D facial keypoint or geometry estimation (e.g.,
[MediaPipe Face Mesh](./face_mesh.md)), facial features or expression
classification, and face region segmentation. BlazeFace uses a lightweight
feature extraction network inspired by, but distinct from
[MobileNetV1/V2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html),
a GPU-friendly anchor scheme modified from
[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325), and an
improved tie resolution strategy alternative to non-maximum suppression. For
more information about BlazeFace, please see the [Resources](#resources)
section.

![face_detection_android_gpu.gif](../images/mobile/face_detection_android_gpu.gif)

## Example Apps

Please first see general instructions for
[Android](../getting_started/building_examples.md#android), [iOS](../getting_started/building_examples.md#ios)
and [desktop](../getting_started/building_examples.md#desktop) on how to build MediaPipe
examples.

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

### Mobile

#### GPU Pipeline

*   Graph:
    [`mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1DZTCy1gp238kkMnu4fUkwI3IrF77Mhy5)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectiongpu:facedetectiongpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectiongpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/facedetectiongpu:FaceDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/facedetectiongpu/BUILD)

#### CPU Pipeline

This is very similar to the [GPU pipeline](#gpu-pipeline) except that at the
beginning and the end of the pipeline it performs GPU-to-CPU and CPU-to-GPU
image transfer respectively. As a result, the rest of graph, which shares the
same configuration as the GPU pipeline, runs entirely on CPU.

*   Graph:
    [`mediapipe/graphs/face_detection/face_detection_mobile_cpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_cpu.pbtxt)
*   Android target:
    [(or download prebuilt ARM64 APK)](https://drive.google.com/open?id=1npiZY47jbO5m2YaL63o5QoCQs40JC6C7)
    [`mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu:facedetectioncpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/android/src/java/com/google/mediapipe/apps/facedetectioncpu/BUILD)
*   iOS target:
    [`mediapipe/examples/ios/facedetectioncpu:FaceDetectionCpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/facedetectioncpu/BUILD)

### Desktop

*   Running on CPU:
    *   Graph:
        [`mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_detection:face_detection_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_detection/BUILD)
*   Running on GPU
    *   Graph:
        [`mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt)
    *   Target:
        [`mediapipe/examples/desktop/face_detection:face_detection_gpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/face_detection/BUILD)

### Web

Please refer to [these instructions](../index.md#mediapipe-on-the-web).

### Coral

Please refer to
[these instructions](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/README.md)
to cross-compile and run MediaPipe examples on the
[Coral Dev Board](https://coral.ai/products/dev-board).

## Resources

*   Paper:
    [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/abs/1907.05047)
    ([presentation](https://docs.google.com/presentation/d/1YCtASfnYyZtH-41QvnW5iZxELFnf0MF-pPWSLGj8yjQ/present?slide=id.g5bc8aeffdd_1_0))
    ([poster](https://drive.google.com/file/d/1u6aB6wxDY7X2TmeUUKgFydulNtXkb3pu/view))
*   For front-facing/selfie camera:
    [TFLite model](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_front.tflite),
    [TFLite model quantized for EdgeTPU/Coral](https://github.com/google/mediapipe/tree/master/mediapipe/examples/coral/models/face-detector-quantized_edgetpu.tflite)
*   For back-facing camera:
    [TFLite model ](https://github.com/google/mediapipe/tree/master/mediapipe/models/face_detection_back.tflite)
*   [Model card](https://drive.google.com/file/d/1f39lSzU5Oq-j_OXgS67KfN5wNsoeAZ4V/view)
