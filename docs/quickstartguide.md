# Quick start guide
- Building docker container with dependencies
```bash
git clone https://github.com/openvinotoolkit/mediapipe.git
cd mediapipe
make docker_build
```
- You can check the integrity of the built image by running tests
```bash
make tests
```

- Running demo applications inside mediapipe_ovms container
```bash
docker run -it mediapipe_ovms:latest bash
```

- Running object detection demo inside mediapipe_ovms container
```bash
make run_object_detection
```

- Running holistic tracking demo inside mediapipe_ovms container
```bash
make run_holistic_tracking
```

- Running iris tracking demo inside mediapipe_ovms container
```bash
make run_iris_tracking
```

- Running pose tracking demo inside mediapipe_ovms container
```bash
make run_pose_tracking
```

- Running face detection demo inside mediapipe_ovms container
```bash
make run_face_detection
```