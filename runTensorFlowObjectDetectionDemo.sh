 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_tensorflow \
	 --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_tensorflow_to_screen_graph.pbtxt \
	--input_side_packets=input_video_path=udp://0.0.0.0:5000
