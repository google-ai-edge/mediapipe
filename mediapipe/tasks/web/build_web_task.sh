#!/bin/bash

# Usage:  
# From mediapipe root folder:
# $ bash ./mediapipe/tasks/web/build_web_task.sh <TASK_NAME>
# TASK_NAME should be one of: common, vision, text, audio, genai

set -e

# Validate and set TASK_NAME
TASK_NAME=$1

if [[ -z "$TASK_NAME" ]]; then
    echo "Error: TASK_NAME is required."
    echo "Usage: $0 TASK_NAME"
    echo "TASK_NAME should be one of: common, vision, text, audio, genai"
    exit 1
fi

# Step 1: Build the Docker image
docker build -t mediapipe .

# Step 2: Define the build script to run inside the container
script="bazel build //mediapipe/tasks/web/$TASK_NAME:all; \
     npm install -g @microsoft/api-extractor; \
     cp mediapipe/tasks/web/$TASK_NAME/api-extractor.json bazel-out/k8-fastbuild/bin/mediapipe/tasks/web/$TASK_NAME/api-extractor.json; \
     cp tsconfig.json bazel-out/k8-fastbuild/bin/mediapipe/tasks/web/$TASK_NAME/tsconfig.json; \
     cd bazel-out/k8-fastbuild/bin/mediapipe/tasks/web/$TASK_NAME; npx api-extractor run; \
     cp -rf ${TASK_NAME}_pkg /mediapipe/${TASK_NAME}_pkg; \
     echo Done. "

# Step 3: Run the container with your local project mounted
docker run -v "$(pwd)":/mediapipe --rm -it mediapipe bash -c "$script"
