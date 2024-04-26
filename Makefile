#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

HTTP_PROXY := "$(http_proxy)"
HTTPS_PROXY := "$(https_proxy)"
OVMS_MEDIA_DOCKER_IMAGE ?= mediapipe_ovms
OVMS_MEDIA_IMAGE_TAG ?= latest
INPUT_VIDEO_LINK ?= "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"
# Main at Fix building without MediaPipe (#2129)
OVMS_COMMIT ?="9bb7942622d30a3272128db03f5e8b158ee81dcc"
JOBS ?= $(shell python3 -c 'import multiprocessing as mp; print(mp.cpu_count())')
DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.1/linux/l_openvino_toolkit_ubuntu20_2024.1.0.15008.f4afc983258_x86_64.tgz

# Targets to use outside running mediapipe_ovms container
docker_build:
	docker build -f Dockerfile.openvino \
	--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTPS_PROXY) \
	--build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL) \
	--build-arg JOBS=$(JOBS) . \
	--build-arg OVMS_COMMIT=$(OVMS_COMMIT) \
	-t $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG)

tests: run_unit_tests run_hello_world run_hello_ovms
run_hello_ovms:
	docker run $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel-bin/mediapipe/examples/desktop/hello_ovms/hello_ovms | grep -q "Output tensor data: 9 - 11"

MEDIAPIPE_UNSTABLE_TESTS_REGEX="MuxInputStreamHandlerTest.RemovesUnusedDataStreamPackets"
run_unit_tests:
	docker run -e http_proxy=$(HTTP_PROXY) -e https_proxy=$(HTTPS_PROXY) $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel test --define=MEDIAPIPE_DISABLE_GPU=1 --test_output=streamed --test_filter="-${MEDIAPIPE_UNSTABLE_TESTS_REGEX}" //mediapipe/framework/...
	docker run -e http_proxy=$(HTTP_PROXY) -e https_proxy=$(HTTPS_PROXY) $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel test --define=MEDIAPIPE_DISABLE_GPU=1 --test_output=streamed //mediapipe/calculators/ovms:all
	docker run -e http_proxy=$(HTTP_PROXY) -e https_proxy=$(HTTPS_PROXY) $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel test --define=MEDIAPIPE_DISABLE_GPU=1 --test_output=streamed //mediapipe/calculators/geti/serialization:all

run_hello_world:
	docker run $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel-bin/mediapipe/examples/desktop/hello_world/hello_world

run_demos_in_docker:
	docker run $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) make run_demos 2>&1 | tee test_demos.log 
	cat test_demos.log | grep -a FPS | grep -v echo
	if [ `cat test_demos.log | grep -a FPS: | wc -l` != "5" ]; then echo "Some demo was not executed correctly. Check the logs"; fi

	# report error if performance reported for less then 5 demos
	cat test_demos.log | grep -a FPS: | wc -l | grep -q "5"

run_python_demos_in_docker:
	docker run $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) make run_python_demos

# Targets to use inside running mediapipe_ovms container
run_demos: run_holistic_tracking run_face_detection run_iris_tracking run_object_detection run_pose_tracking

run_python_demos: run_python_object_detection run_python_holistic_tracking run_python_face_detection

run_object_detection:
	echo "Running FPS test for object_detection demo"
	rm -rf /mediapipe/output_object_detection_ovms.mp4
	bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_ovms --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_ovms_graph.pbtxt --input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4 --output_video_path=/mediapipe/object_detection_ovms.mp4
	
run_holistic_tracking:
	echo "Running FPS test for holistic_tracking demo"
	rm -rf /mediapipe/output_holistic_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_holistic_ovms.mp4

run_face_detection:
	echo "Running FPS test for face_detection demo"
	rm -rf /mediapipe/output_face_detection_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_face_detection_ovms.mp4

run_iris_tracking:
	echo "Running FPS test for iris_tracking demo"
	rm -rf /mediapipe/output_iris_tracking_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_iris_tracking_ovms.mp4

run_pose_tracking:
	echo "Running FPS test for pose_tracking demo"
	rm -rf /mediapipe/output_pose_track_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_pose_track_ovms.mp4

run_python_object_detection:
	echo "Running python ovms object detection demo"
	cp build/lib.linux-x86_64-cpython-38/mediapipe/examples/python/ovms_object_detection.py build/lib.linux-x86_64-cpython-38
	python build/lib.linux-x86_64-cpython-38/ovms_object_detection.py

run_python_holistic_tracking:
	echo "Running python ovms holistic tracking demo"
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	cp build/lib.linux-x86_64-cpython-38/mediapipe/examples/python/ovms_holistic_tracking.py build/lib.linux-x86_64-cpython-38
	python build/lib.linux-x86_64-cpython-38/ovms_holistic_tracking.py

run_python_face_detection:
	echo "Running python ovms face detection demo"
	if [ ! -f video.mp4 ]; then wget -O video.mp4 $(INPUT_VIDEO_LINK); fi
	cp build/lib.linux-x86_64-cpython-38/mediapipe/examples/python/ovms_face_detection.py build/lib.linux-x86_64-cpython-38
	python build/lib.linux-x86_64-cpython-38/ovms_face_detection.py

