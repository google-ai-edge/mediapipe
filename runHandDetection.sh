bazel-bin/mediapipe/examples/desktop/hand_detection/hand_detection_tflite \
    --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_detection_desktop.pbtxt \
    --input_side_packets=input_video_path=rtp://0.0.0.0:5000
