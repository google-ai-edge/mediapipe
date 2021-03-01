---
layout: default
title: Object Classification
parent: Solutions
nav_order: TODO
---

# MediaPipe Object Classification
{: .no_toc }

1. TOC
{:toc}
---

## Example Apps

Note: To visualize a graph, copy the graph and paste it into
[MediaPipe Visualizer](https://viz.mediapipe.dev/). For more information on how
to visualize its associated subgraphs, please see
[visualizer documentation](../tools/visualizer.md).

<!-- ### Mobile

Please first see general instructions for
[iOS](../getting_started/building_examples.md#ios) on how to build MediaPipe examples.

#### GPU Pipeline

*   iOS target:
    [`mediapipe/examples/ios/objectdetectiongpu:ObjectDetectionGpuApp`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/ios/objectdetectiongpu/BUILD)
 -->

### Desktop

#### Live Camera Input

Please first see general instructions for
[desktop](../getting_started/building_examples.md#desktop) on how to build MediaPipe examples.

*   Graph:
    [`mediapipe/graphs/object_classification/object_classification_desktop_live.pbtxt`](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_classification/object_classification_desktop_live.pbtxt)
*   Target:
    [`mediapipe/examples/desktop/object_classification:object_classification_pytorch_cpu`](https://github.com/google/mediapipe/tree/master/mediapipe/examples/desktop/object_classification/BUILD)

#### Video File Input

*   With a PyTorch Model

    This uses a MobileNetv2 trace model from PyTorch Hub. To fetch and prepare it, run:

    ```bash
    python mediapipe/models/trace_mobilenetv2.py
    ```

    The pipeline is implemented in this
    [graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/object_classification/object_classification_desktop_live.pbtxt).

    To build the application, run:

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_classification:object_classification_pytorch_cpu
    ```

    To run the application, replace `<input video path>` and `<output video
    path>` in the command below with your own paths:

    Tip: You can find a test video available in
    `mediapipe/examples/desktop/object_detection`.

    ```
    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_classification/object_classification_pytorch_cpu \
      --calculator_graph_config_file=mediapipe/graphs/object_classification/object_classification_desktop_live.pbtxt \
      --input_side_packets=input_video_path=<input video path>,output_video_path=<output video path>
    ```
<!-- 
## Resources

*   [Models and model cards](./models.md#object_detection)
 -->