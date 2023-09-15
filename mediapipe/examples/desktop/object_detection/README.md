# Object detection demo

OpenVINO&trade; Model Server fork of [MediaPipe](https://google.github.io/mediapipe/) repository allowing users to take advantage of OpenVINO&trade; backend in mediapipe framework.

[Original demo documentation](https://google.github.io/mediapipe/solutions/object_detection)

# Building docker container with dependencies
```bash
git clone https://github.com/openvinotoolkit/mediapipe.git
cd mediapipe
make docker_build
```

# Start mediapipe_ovms container
```bash
docker run -it mediapipe_ovms:latest bash
```

# Running demo inside mediapipe_ovms container

## Prepare models and build application
```bash
python setup_ovms.py --get_models
build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_ovms
```

## Download the input video or prepare your own input as 'video.mp4'
```bash
wget -O video.mp4 "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"
```

## Run the demo
Execute command with the calculator_graph_config_file, input_video_path and output_video_path:
```bash
bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_openvino --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_ovms_graph.pbtxt --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/tested_video.mp4"
```

Now you can review the tested_video.mp4 output with detections.