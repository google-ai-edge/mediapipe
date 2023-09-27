#/bin/bash
make docker_build
docker run -it  mediapipe_ovms:latest make run_demos | tee test_demos.log
cat test_demos.log | grep -a FPS | grep -v echo
