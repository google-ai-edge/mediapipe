**Hello World**

To build the "Hello World" example, use:

```
bazel build -c opt mediapipe/examples/desktop/hello_world:hello_world
```

and then run it using:

```
bazel-bin/mediapipe/examples/desktop/hello_world/hello_world --logtostderr
```

**TFlite Object Detection**

To build the object detection demo using a TFLite model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/object_detection:object_detection_tflite --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file \
  --alsologtostderr
```

**TensorFlow Object Detection**

To build the object detection demo using a TensorFlow model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/object_detection:object_detection_tensorflow \
  --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tensorflow  \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_graph.pbtxt  \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
  --alsologtostderr
```
