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
OVMS_BRANCH ?= "mediapipe_integration"
JOBS ?= $(shell python3 -c 'import multiprocessing as mp; print(mp.cpu_count())')
DLDT_PACKAGE_URL ?= https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.0/linux/l_openvino_toolkit_ubuntu20_2023.0.0.10926.b4452d56304_x86_64.tgz

# Targets to use outside running mediapipe_ovms container
docker_build:
	docker build -f Dockerfile.openvino \
	--build-arg http_proxy=$(HTTP_PROXY) --build-arg https_proxy=$(HTTPS_PROXY) \
	--build-arg DLDT_PACKAGE_URL=$(DLDT_PACKAGE_URL) \
	--build-arg JOBS=$(JOBS) . \
	--build-arg OVMS_BRANCH=$(OVMS_BRANCH) \
	-t $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG)

tests: run_unit_tests run_hello_world run_hello_ovms
run_hello_ovms:
	docker run -it $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel-bin/mediapipe/examples/desktop/hello_ovms/hello_ovms | grep "Output tensor data: 9 - 11"

run_unit_tests:
	docker run -it $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel test --define=MEDIAPIPE_DISABLE_GPU=1 //mediapipe/framework/... | grep "Build completed successfully"

run_hello_world:
	docker run -it $(OVMS_MEDIA_DOCKER_IMAGE):$(OVMS_MEDIA_IMAGE_TAG) bazel-bin/mediapipe/examples/desktop/hello_world/hello_world | grep "Hello World!"

# Targets to use inside running mediapipe_ovms container
run_object_detection:
	python setup_ovms.py --get_models
	python setup_ovms.py --convert_pose
	bash build_desktop_examples.sh -t object_detection
	bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_ovms --calculator_graph_config_file mediapipe/graphs/object_detection/object_detection_desktop_ovms1_graph.pbtxt --input_side_packets "input_video_path=/mediapipe/mediapipe/examples/desktop/object_detection/test_video.mp4,output_video_path=/mediapipe/output_object_detection_ovms.mp4"
	
run_holistic_tracking:
	rm -rf /mediapipe/output_holistic_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"; fi
	python setup_ovms.py --get_models
	python setup_ovms.py --convert_pose
	bash build_desktop_examples.sh -t holistic_tracking
	bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_holistic_ovms.mp4

run_face_detection:
	rm -rf /mediapipe/output_face_detection_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"; fi
	python setup_ovms.py --get_models
	python setup_ovms.py --convert_pose
	bash build_desktop_examples.sh -t face_detection
	bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_face_detection_ovms.mp4

run_iris_tracking:
	rm -rf /mediapipe/output_iris_tracking_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"; fi
	python setup_ovms.py --get_models
	python setup_ovms.py --convert_pose
	bash build_desktop_examples.sh -t iris_tracking
	bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_iris_tracking_ovms.mp4

run_pose_tracking:
	rm -rf /mediapipe/output_pose_track_ovms.mp4
	if [ ! -f video.mp4 ]; then wget -O video.mp4 "https://www.pexels.com/download/video/3044127/?fps=24.0&h=1080&w=1920"; fi
	python setup_ovms.py --get_models
	python setup_ovms.py --convert_pose
	bash build_desktop_examples.sh -t pose_tracking
	bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu --calculator_graph_config_file /mediapipe/mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt --input_video_path=/mediapipe/video.mp4 --output_video_path=/mediapipe/output_pose_track_ovms.mp4

