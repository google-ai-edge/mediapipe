#!/usr/bin/env bash
# Copyright 2023 The MediaPipe Authors. All Rights Reserved.
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

# Set the following variables as appropriate.
#   * BAZEL: path to bazel. defaults to the first one available in PATH
#   * FRAMEWORK_NAME: name of the iOS framework to be built. Currently the
#   * accepted values are MediaPipeTasksCommon, MediaPipeTasksText, MediaPipeTasksVision.
#   * MPP_BUILD_VERSION: to specify the release version. defaults to 0.0.1-dev
#   * IS_RELEASE_BUILD: set as true if this build should be a release build
#   * ARCHIVE_FRAMEWORK: set as true if the framework should be archived
#   * DEST_DIR: destination directory to which the framework will be copied.

set -ex

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This build script only works on macOS."
  exit 1
fi

BAZEL="${BAZEL:-$(which bazel)}"
MPP_BUILD_VERSION=${MPP_BUILD_VERSION:-0.0.1-dev}
MPP_ROOT_DIR=$(git rev-parse --show-toplevel)
ARCHIVE_FRAMEWORK=${ARCHIVE_FRAMEWORK:-true}
IS_RELEASE_BUILD=${IS_RELEASE_BUILD:-false}
DEST_DIR=${DEST_DIR:-$HOME}

echo "Destination"
echo "${DEST_DIR}"

if [[ ! -x "${BAZEL}" ]]; then
  echo "bazel executable is not found."
  exit 1
fi

if [ -z ${FRAMEWORK_NAME+x} ]; then
  echo "Name of the iOS framework, which is to be built, must be set."
  exit 1
fi

case $FRAMEWORK_NAME in
  "MediaPipeTasksCommon")
    ;;
  "MediaPipeTasksVision")
    ;;
  "MediaPipeTasksText")
    ;;
  "MediaPipeTasksGenAIC")
    ;;
  "MediaPipeTasksGenAI")
    ;;
  *)
    echo "Wrong framework name. The following framework names are allowed: MediaPipeTasksText, MediaPipeTasksVision, MediaPipeTasksCommon, MediaPipeTasksGenAI, MediaPipeTasksGenAIC"
    exit 1
  ;;
esac

if [[ -z "${DEST_DIR+x}" || "${DEST_DIR}" == ${MPP_ROOT_DIR}* ]]; then
  echo "DEST_DIR variable must be set and not be under the repository root."
  exit 1
fi

# This function takes one bazel target label as an argument, and prints
# the path of the first output file of the specified target.
function get_output_file_path {
  local STARLARK_OUTPUT_TMPDIR="$(mktemp -d)"

  local STARLARK_FILE="${STARLARK_OUTPUT_TMPDIR}/print_output_file.starlark"
  cat > "${STARLARK_FILE}" << EOF
def format(target):
  return target.files.to_list()[0].path
EOF

  local OUTPUT_PATH=$(bazel cquery $1 --output=starlark --starlark:file="${STARLARK_FILE}" 2> /dev/null)

  rm -rf "${STARLARK_OUTPUT_TMPDIR}"

  echo ${OUTPUT_PATH}
}

# This function builds 3 the xcframework and associated graph libraries if any
# for a given framework name.
function build_ios_frameworks_and_libraries {
  local TARGET_PREFIX="//mediapipe/tasks/ios"
  FULL_FRAMEWORK_TARGET="${TARGET_PREFIX}:${FRAMEWORK_NAME}_framework"

  # .bazelrc sets --apple_generate_dsym=true by default which bloats the libraries to sizes of
  # the order of GBs. All iOS framework and library build commands for distribution via
  # CocoaPods must set --apple_generate_dsym=false inorder to shave down the binary size to
  # the order of a few MBs.

  # Build Task Library xcframework.
  local FRAMEWORK_CQUERY_COMMAND="-c opt --config=ios_sim_device_fat --apple_generate_dsym=false --define OPENCV=source ${FULL_FRAMEWORK_TARGET}"

  ${BAZEL} build ${FRAMEWORK_CQUERY_COMMAND}
  IOS_FRAMEWORK_PATH="$(get_output_file_path "${FRAMEWORK_CQUERY_COMMAND}")"

  case $FRAMEWORK_NAME in
    # `MediaPipeTasksCommon` pods must also include the task graph libraries which
  # are to be force loaded. Hence the graph libraies are only built if the framework
  # name is `MediaPipeTasksCommon`.`
    "MediaPipeTasksCommon")
      local IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND="-c opt --config=ios_sim_fat --apple_generate_dsym=false --define OPENCV=source //mediapipe/tasks/ios:MediaPipeTaskGraphs_library"
      ${BAZEL} build ${IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND}
      IOS_GRAPHS_SIMULATOR_LIBRARY_PATH="$(get_output_file_path "${IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND}")"

      # Build static library for iOS devices with arch ios_arm64. We don't need to build for armv7 since
      # our deployment target is iOS 12.0. iOS 12.0 and upwards is not supported by old armv7 devices.
      local IOS_DEVICE_LIBRARY_CQUERY_COMMAND="-c opt --config=ios_arm64 --apple_generate_dsym=false --define OPENCV=source //mediapipe/tasks/ios:MediaPipeTaskGraphs_library"
      ${BAZEL} build ${IOS_DEVICE_LIBRARY_CQUERY_COMMAND}
      IOS_GRAPHS_DEVICE_LIBRARY_PATH="$(get_output_file_path "${IOS_DEVICE_LIBRARY_CQUERY_COMMAND}")"
      ;;
    # This section is for internal purposes only.
    "MediaPipeTasksGenAIC")
      if [[ ! -z ${ENABLE_ODML_COCOAPODS_BUILD+x} ]]; then
        local IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND="-c opt --config=ios_sim_fat --apple_generate_dsym=false //mediapipe/tasks/ios:MediaPipeTasksGenAI_library"
        ${BAZEL} build ${IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND}
        IOS_GENAI_SIMULATOR_LIBRARY_PATH="$(get_output_file_path "${IOS_SIM_FAT_LIBRARY_CQUERY_COMMAND}")"

        # Build static library for iOS devices with arch ios_arm64. We don't need to build for armv7 since
        # our deployment target is iOS 12.0. iOS 12.0 and upwards is not supported by old armv7 devices.
        local IOS_DEVICE_LIBRARY_CQUERY_COMMAND="-c opt --config=ios_arm64 --apple_generate_dsym=false //mediapipe/tasks/ios:MediaPipeTasksGenAI_library"
        ${BAZEL} build ${IOS_DEVICE_LIBRARY_CQUERY_COMMAND}
        IOS_GENAI_DEVICE_LIBRARY_PATH="$(get_output_file_path "${IOS_DEVICE_LIBRARY_CQUERY_COMMAND}")"
      fi
      ;;
    *)
      ;;
  esac
}

function create_framework_archive {
  # Change to the Bazel iOS output directory.
  pushd "${MPP_ROOT_DIR}"

  # Create the temporary directory for the given framework.
  local ARCHIVE_NAME="${FRAMEWORK_NAME}-${MPP_BUILD_VERSION}"
  local MPP_TMPDIR="$(mktemp -d)"

  # Copy the license file to MPP_TMPDIR
  cp "LICENSE" ${MPP_TMPDIR}

  # Unzip the iOS framework zip generated by bazel to MPP_TMPDIR
  local FRAMEWORKS_DIR="${MPP_TMPDIR}/frameworks"

  echo ${IOS_FRAMEWORK_PATH}
  unzip "${IOS_FRAMEWORK_PATH}" -d "${FRAMEWORKS_DIR}"

  case $FRAMEWORK_NAME in
    # If the framework being built is `MediaPipeTasksCommon`, the built graph
    # libraries should be copied to the output directory which is to be archived.
    "MediaPipeTasksCommon")
      local GRAPH_LIBRARIES_DIR="graph_libraries"
      # Create the parent folder which will hold the graph libraries of all architectures.
      mkdir -p "${FRAMEWORKS_DIR}/${GRAPH_LIBRARIES_DIR}"

      local SIMULATOR_GRAPH_LIBRARY_PATH="${FRAMEWORKS_DIR}/${GRAPH_LIBRARIES_DIR}/lib${FRAMEWORK_NAME}_simulator_graph.a"

      # Copy ios simulator fat library into a separate directory.
      echo ${IOS_GRAPHS_SIMULATOR_LIBRARY_PATH}
      cp "${IOS_GRAPHS_SIMULATOR_LIBRARY_PATH}" "${SIMULATOR_GRAPH_LIBRARY_PATH}"

      local IOS_DEVICE_GRAPH_LIBRARY_PATH="${FRAMEWORKS_DIR}/${GRAPH_LIBRARIES_DIR}/lib${FRAMEWORK_NAME}_device_graph.a"

      # Copy ios device library into a separate directory.
      echo ${IOS_GRAPHS_DEVICE_LIBRARY_PATH}
      cp "${IOS_GRAPHS_DEVICE_LIBRARY_PATH}" "${IOS_DEVICE_GRAPH_LIBRARY_PATH}"
      ;;
    # This section is for internal purposes only.
    "MediaPipeTasksGenAIC")
      if [[ ! -z ${ENABLE_ODML_COCOAPODS_BUILD+x} ]]; then
        local GENAI_LIBRARIES_DIR="genai_libraries"
        # Create the parent folder which will hold the genai libraries of all architectures.
        mkdir -p "${FRAMEWORKS_DIR}/${GENAI_LIBRARIES_DIR}"

        local SIMULATOR_GENAI_LIBRARY_PATH="${FRAMEWORKS_DIR}/${GENAI_LIBRARIES_DIR}/lib${FRAMEWORK_NAME}_simulator.a"

        # Copy ios simulator fat library into a separate directory.
        echo ${IOS_GENAI_SIMULATOR_LIBRARY_PATH}
        cp "${IOS_GENAI_SIMULATOR_LIBRARY_PATH}" "${SIMULATOR_GENAI_LIBRARY_PATH}"

        local IOS_DEVICE_GENAI_LIBRARY_PATH="${FRAMEWORKS_DIR}/${GENAI_LIBRARIES_DIR}/lib${FRAMEWORK_NAME}_device.a"

        # Copy ios device library into a separate directory.
        echo ${IOS_GENAI_DEVICE_LIBRARY_PATH}
        cp "${IOS_GENAI_DEVICE_LIBRARY_PATH}" "${IOS_DEVICE_GENAI_LIBRARY_PATH}"
      fi
      ;;
    *)
      ;;
  esac

  #----- (3) Move the framework to the destination -----
  if [[ "${ARCHIVE_FRAMEWORK}" == true ]]; then
    # Create the framework archive directory.
    mkdir -p "${FRAMEWORK_NAME}"
    local TARGET_DIR="$(realpath "${FRAMEWORK_NAME}")"

    local FRAMEWORK_ARCHIVE_DIR
    if [[ "${IS_RELEASE_BUILD}" == true ]]; then
      # Get the first 16 bytes of the sha256 checksum of the root directory.
      local SHA256_CHECKSUM=$(find "${MPP_TMPDIR}" -type f -print0 | xargs -0 shasum -a 256 | sort | shasum -a 256 | cut -c1-16)
      FRAMEWORK_ARCHIVE_DIR="${TARGET_DIR}/${MPP_BUILD_VERSION}/${SHA256_CHECKSUM}"
    else
      FRAMEWORK_ARCHIVE_DIR="${TARGET_DIR}/${MPP_BUILD_VERSION}"
    fi
    mkdir -p "${FRAMEWORK_ARCHIVE_DIR}"

    # Zip up the framework and move to the archive directory.
    pushd "${MPP_TMPDIR}"
    local MPP_ARCHIVE_FILE="${ARCHIVE_NAME}.tar.gz"
    tar -cvzf "${MPP_ARCHIVE_FILE}" .
    mv "${MPP_ARCHIVE_FILE}" "${FRAMEWORK_ARCHIVE_DIR}"
    popd

    # Move the target directory to the Kokoro artifacts directory and clean up
    # the artifacts directory in the mediapipe root directory even if the
    # move command fails.
    mv "${TARGET_DIR}" "$(realpath "${DEST_DIR}")"/ || true
    rm -rf "${TARGET_DIR}"
  else
    rsync -r "${MPP_TMPDIR}/" "$(realpath "${DEST_DIR}")/"
  fi

  # Clean up the temporary directory for the framework.
  rm -rf "${MPP_TMPDIR}"
  echo ${MPP_TMPDIR}
}

cd "${MPP_ROOT_DIR}"
build_ios_frameworks_and_libraries
create_framework_archive
