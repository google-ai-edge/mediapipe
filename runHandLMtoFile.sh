bazel-bin/myMediapipe/projects/staticGestures/staticGesturesCaptureToFile/static_gestures_cpu_tflite \
    --calculator_graph_config_file=myMediapipe/graphs/staticGesturesCaptureToFile/mainGraph_desktop.pbtxt \
    --input_side_packets=input_video_path=rtp://0.0.0.0:5000
