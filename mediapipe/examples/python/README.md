# OVMS python examples
- Building docker container with dependencies
```bash
git clone https://github.com/openvinotoolkit/mediapipe.git
cd mediapipe
make docker_build
```

- Start the container
```bash
docker run -it mediapipe_ovms:latest bash
```

# Object Detection
- Run example ovms python script for object detection - whole input video as one execution
```bash
cp mediapipe/examples/python/ovms_object_detection.py build/lib.linux-x86_64-cpython-38/
python build/lib.linux-x86_64-cpython-38/ovms_object_detection.py
```

This execution will produce object_output.mp4 vide with object detection results found in the input video file:
```bash
/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4
```

You can manipulate the input and output video file paths with command line arguments:
```bash
python build/lib.linux-x86_64-cpython-38/ovms_object_detection.py --input_video_path /my/path --output_video_path /my/output/path
```

- This script will run object detection on input video, as described in this c++ example
[OVMS Object Detection](../desktop/object_detection/README.md)
[Original demo documentation](https://google.github.io/mediapipe/solutions/object_detection)

# Holistic Tracking
- Run example ovms python script for holistic tracking - frame by frame execution
```bash
cp mediapipe/examples/python/ovms_holistic_tracking.py build/lib.linux-x86_64-cpython-38/
python build/lib.linux-x86_64-cpython-38/ovms_holistic_tracking.py
```

This execution will produce holistic_output.mp4 vide with object detection results found in the input video file:
```bash
/mediapipe/video.mp4
```

You can manipulate the input and output video file paths used to execute the graph with command line arguments:
```bash
python build/lib.linux-x86_64-cpython-38/ovms_holistic_tracking.py --input_video_path /my/path --output_video_path /my/output/path
```

- This script will run holistic tracking on input video, frame by frame as described in this c++ example
[OVMS Object Detection](../desktop/holistic_tracking/README.md)
[Original demo documentation](https://google.github.io/mediapipe/solutions/holistic)

# Holistic Tracking Multithread
- Run example ovms python script for holistic tracking - frame by frame execution
```bash
cp mediapipe/examples/python/ovms_holistic_tracking_multithread.py build/lib.linux-x86_64-cpython-38/
python build/lib.linux-x86_64-cpython-38/ovms_holistic_tracking_multithread.py
```

This execution will produce holistic_output.mp4 vide with object detection results found in the input video file:
```bash
/mediapipe/video.mp4
```

You can manipulate the input and output video file paths as well as the number of threads used to execute the graph with command line arguments:
```bash
python build/lib.linux-x86_64-cpython-38/ovms_holistic_tracking_multithread.py --input_video_path /my/path --output_video_path /my/output/path --num_threads 2
```

# Face Detection
- Run example ovms python script for face detection - frame by frame execution
```bash
cp mediapipe/examples/python/ovms_face_detection.py build/lib.linux-x86_64-cpython-38/
python build/lib.linux-x86_64-cpython-38/ovms_face_detection.py
```

This execution will produce face_output.mp4 vide with object detection results found in the input video file:
```bash
/mediapipe/video.mp4
```

You can manipulate the input and output video file paths used to execute the graph with command line arguments:
```bash
python build/lib.linux-x86_64-cpython-38/ovms_face_detection.py --input_video_path /my/path --output_video_path /my/output/path
```

- This script will run face detection on input video, frame by frame as described in this c++ example
[OVMS Object Detection](../desktop/face_detection/README.md)
[Original demo documentation](https://google.github.io/mediapipe/solutions/face_detection)