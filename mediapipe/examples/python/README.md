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

- Prepare models for ovms
```bash
python setup_ovms.py --get_models'
```

- Build and install mediapipe python package
Make sure you are in /mediapipe dirctory
Below command takes around 1 hour depending on your internet speed and cpu
```bash
pip install .
```

- Run example ovms python script
```bash
python build/lib.linux-x86_64-cpython-38/mediapipe/examples/python/ovms_object_detection.py
```

- This script will run object detection on input video, as described in this c++ example
[OVMS Object Detection](../desktop/object_detection/README.md)
[Original demo documentation](https://google.github.io/mediapipe/solutions/object_detection)