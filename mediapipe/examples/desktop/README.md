**Hello World**

To build the "Hello World" example, use:

```
bazel build -c opt mediapipe/examples/desktop/hello_world:hello_world
```

and then run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/hello_world/hello_world
```

**TFlite Object Detection**

To build the object detection demo using a TFLite model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/object_detection:object_detection_tflite --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tflite \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tflite_graph.pbtxt \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
```

**TensorFlow Object Detection**

To build the object detection demo using a TensorFlow model on desktop, use:

```
export GLOG_logtostderr=1

bazel build -c opt mediapipe/examples/desktop/object_detection:object_detection_tensorflow \
  --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tensorflow  \
  --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_graph.pbtxt  \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
```

**TFlite Hand Detection**

To build the hand detection demo using a TFLite model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/hand_tracking:hand_tracking_tflite --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_tflite \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_detection_desktop.pbtxt \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
```

**TFlite Hand Tracking**

To build the hand tracking demo using a TFLite model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/hand_tracking:hand_tracking_tflite --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_tflite \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop.pbtxt \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
```

**TFlite Multi-Hand Tracking**

To build the multi-hand tracking demo using a TFLite model on desktop, use:

```
bazel build -c opt mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_tflite --define MEDIAPIPE_DISABLE_GPU=1
```

and run it using:

```
export GLOG_logtostderr=1

bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_tflite \
  --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop.pbtxt \
  --input_side_packets=input_video_path=/path/to/input/file,output_video_path=/path/to/output/file
```

To change the number of hands to `x` in this application, change:

1. `min_size:x` in `CollectionHasMinSizeCalculatorOptions` in `mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop.pbtxt`.
2. `max_vec_size:x` in `ClipVectorSizeCalculatorOptions` in `mediapipe/examples/dekstop/hand_tracking/subgraphs/multi_hand_detection_cpu.pbtxt`.
