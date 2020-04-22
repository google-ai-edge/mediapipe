# Template Matching using KNIFT on Desktop

This doc focuses on the
[example graph](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/template_matching/template_matching_desktop.pbtxt)
that performs template matching with KNIFT (Keypoint Neural Invariant Feature
Transform) on desktop CPU.

If you are interested in more detail about KNIFT or running the example on
mobile, please see
[Template Matching using KNIFT on Mobile (CPU)](template_matching_mobile_cpu.md).

To build the desktop app, run:

```bash
$ bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/template_matching:template_matching_tflite
```

To run the desktop app, please specify a template index file
([example](https://github.com/google/mediapipe/tree/master/mediapipe/models/knift_index.pb)) and a
video to be matched. For how to build your own index file, please see
[here](template_matching_mobile_cpu.md#build-index-file).

```bash
$ GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/template_matching/template_matching_tflite \
    --calculator_graph_config_file=mediapipe/graphs/template_matching/template_matching_desktop.pbtxt --input_side_packets="input_video_path=<input video path>,output_video_path=<output video path>"
```

## Graph

[Source pbtxt file](https://github.com/google/mediapipe/tree/master/mediapipe/graphs/template_matching/template_matching_desktop.pbtxt)
