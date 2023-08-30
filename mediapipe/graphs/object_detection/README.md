# OpenVINO&trade; Model Server fork of [MediaPipe](https://google.github.io/mediapipe/) repository allowing users to take advantage of OpenVINO&trade; Model Serving in mediapipe examples.

# Building docker container with dependencies
```bash
git clone https://github.com/openvinotoolkit/mediapipe.git
cd mediapipe
make docker_build
```
# You can check the integrity of the built image by running tests
```bash
make tests
```

# Running demo applications inside mediapipe_ovms container
```bash
docker run -it mediapipe_ovms:latest bash
```

# Running object detection demo on ovms inside mediapipe_ovms container 
```bash
make run_object_detection
```

# Running object detection demo on ov inside mediapipe_ovms container
```bash
build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_openvino
python setup_ovms.py --get_models
bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_openvino --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_openvino_graph.pbtxt --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/tested_video.mp4"
```