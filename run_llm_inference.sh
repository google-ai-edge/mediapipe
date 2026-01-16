#!/bin/bash

# This is a simple script to run LLM inference on Android via the MediaPipe
# LLM inference engine.
#
# This script allows running transformer-based LLM models in *.task or *.bin
# format. We recommend using `gemma2-2b-it-cpu-int8.task` (from
# https://www.kaggle.com/models/google/gemma-2/tfLite/gemma2-2b-it-cpu-int8) or
# the smaller `gemma-1.1-2b-it-cpu-int4.bin` model (from
# https://www.kaggle.com/models/google/gemma/tfLite/gemma-1.1-2b-it-cpu-int4).

MODEL_FILENAME="gemma2-2b-it-cpu-int8.task"
ADB_WORK_DIR="/data/local/tmp"
INPUT_PROMPT="What is the most famous building in Paris?"

if [ ! -f "${MODEL_FILENAME}" ]; then
  echo "Error: ${MODEL_FILENAME} not found."
  echo "Please download it from https://www.kaggle.com/models/google/gemma-2/tfLite/gemma2-2b-it-cpu-int8"
  exit 1
fi

adb push "${MODEL_FILENAME}" "${ADB_WORK_DIR}/${MODEL_FILENAME}"

# Build the MediaPipe Docker base image.
docker build . --tag=mediapipe

# Build the LLM inference engine binary and copy to the conencted Android device.
CONTAINER_NAME=mediapipe_$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 10 | head -n 1)
docker run --name "$CONTAINER_NAME" mediapipe:latest  sh -c "
    chmod +x  setup_android_sdk_and_ndk.sh && \
    ./setup_android_sdk_and_ndk.sh ~/Android/Sdk ~/Android/Ndk r28b \
        --accept-licenses &&
    bazel build --config android_arm64 --client_env=CC=clang-16 -c opt \
        --copt=-DABSL_FLAGS_STRIP_NAMES=0 \
        --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
        //mediapipe/tasks/cc/genai/inference/c:llm_inference_engine_cpu_main
"
docker cp "$CONTAINER_NAME":/mediapipe/bazel-bin/mediapipe/tasks/cc/genai/inference/c/llm_inference_engine_cpu_main llm_inference_engine_cpu_main
adb push llm_inference_engine_cpu_main "${ADB_WORK_DIR}"/llm_inference_engine_cpu_main

# Run the inference.
adb shell "taskset f0 ${ADB_WORK_DIR}/llm_inference_engine_cpu_main \
              --model_path='${ADB_WORK_DIR}/${MODEL_FILENAME}' \
              --prompt='${INPUT_PROMPT}'"
