#!/usr/bin/env bash
set -euo pipefail

# Builds the manylinux image and extracts wheels into ./wheelhouse.
IMAGE_NAME="${IMAGE_NAME:-mp_manylinux}"
PYTHON_BIN="${PYTHON_BIN:-/opt/python/cp312-cp312/bin/python3.12}"
MEDIAPIPE_PYTHON_BUILD_DIFF_URL="${MEDIAPIPE_PYTHON_BUILD_DIFF_URL:-https://github.com/chromium/chromium/blob/main/third_party/mediapipe/src/third_party/mediapipe_python_build.diff}"
CONTAINER_NAME="${CONTAINER_NAME:-mp_pip_package_container}"

DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.manylinux_2_28_x86_64 \
  -t "${IMAGE_NAME}" . \
  --build-arg "PYTHON_BIN=${PYTHON_BIN}" \
  --build-arg "MEDIAPIPE_PYTHON_BUILD_DIFF_URL=${MEDIAPIPE_PYTHON_BUILD_DIFF_URL}"

docker create -ti --name "${CONTAINER_NAME}" "${IMAGE_NAME}:latest" >/dev/null
mkdir -p wheelhouse
docker cp "${CONTAINER_NAME}:/wheelhouse/." wheelhouse/
docker rm -f "${CONTAINER_NAME}" >/dev/null
