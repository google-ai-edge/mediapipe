### Steps to run the AutoFlip video cropping graph

1.  Checkout the repository and follow
    [the installation instructions](https://github.com/google/mediapipe/blob/master/mediapipe/docs/install.md)
    to set up MediaPipe.

    ```bash
    git clone https://github.com/google/mediapipe.git
    cd mediapipe
    ```

2.  Build and run the run_autoflip binary to process a local video.

Note: AutoFlip currently only works with OpenCV 3 . Please verify your OpenCV version beforehand.

    ```bash
    bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
      mediapipe/examples/desktop/autoflip:run_autoflip

    GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
      --calculator_graph_config_file=mediapipe/examples/desktop/autoflip/autoflip_graph.pbtxt \
      --input_side_packets=input_video_path=/absolute/path/to/the/local/video/file,output_video_path=/absolute/path/to/save/the/output/video/file,aspect_ratio=width:height
    ```

3.  View the cropped video.
