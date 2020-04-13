

$ bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    mediapipe/examples/desktop/simpleIO:simple_io_tflite

# It should print:
# Target //mediapipe/examples/desktop/simpleIO:simple_io_tflite up-to-date:
#   bazel-bin/mediapipe/examples/desktop/simpleIO/simple_io_tflite
# INFO: Elapsed time: 36.417s, Critical Path: 23.22s
# INFO: 711 processes: 710 linux-sandbox, 1 local.
# INFO: Build completed successfully, 734 total actions

$ export GLOG_logtostderr=1

# INPUT=  file, OUTPUT=file
# Replace <input video path> and <output video path>.
# You can find a test video in mediapipe/examples/desktop/simpleIO.
$ bazel-bin/mediapipe/examples/desktop/simpleIO/simple_io_tflite \
    --calculator_graph_config_file=mediapipe/graphs/simple_io/simple_io_graph.pbtxt \
    --input_side_packets=input_video_path=./mediapipe/examples/desktop/simpleIO/test_video.mp4,output_video_path=./mediapipe/examples/desktop/simpleIO/output_video.mp4

# INPUT=  file, OUTPUT=screen

$ bazel-bin/mediapipe/examples/desktop/simpleIO/simple_io_tflite \
    --calculator_graph_config_file=mediapipe/graphs/simple_io/simple_media_to_screen_graph.pbtxt \
    --input_side_packets=input_video_path=./mediapipe/examples/desktop/simpleIO/test_video.mp4


# INPUT=  Stream , OUTPUT=screen

$ bazel-bin/mediapipe/examples/desktop/simpleIO/simple_io_tflite \
    --calculator_graph_config_file=mediapipe/graphs/simple_io/simple_media_to_screen_graph.pbtxt \
    --input_side_packets=input_video_path=rtp://0.0.0.0:5000