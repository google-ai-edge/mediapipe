#!/bin/bash
# Copyright 2020 The MediaPipe Authors.
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
# =========================================================================
#
# Script to build/run all MediaPipe desktop example apps (with webcam input).
#
# To build and run all apps and store them in out_dir:
#   $ ./build_desktop_examples.sh -d out_dir
#   Omitting -d and the associated directory saves all generated apps in the
#   current directory.
# To build all apps and store them in out_dir:
#   $ ./build_desktop_examples.sh -d out_dir -b
#   Omitting -d and the associated directory saves all generated apps in the
#   current directory.
# To run all apps already stored in out_dir:
#   $ ./build_desktop_examples.sh -d out_dir -r
#   Omitting -d and the associated directory assumes all apps are in the current
#   directory.

set -e

out_dir="."
build_only=false
run_only=false
one_target=false
app_dir="mediapipe/examples/desktop"
bin_dir="bazel-bin"
declare -a default_bazel_flags=(build -c opt  --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 --define=MEDIAPIPE_DISABLE=1 --define=PYTHON_DISABLE=1 --cxxopt=-DPYTHON_DISABLE=1 --cxxopt=-DMEDIAPIPE_DISABLE=1 --define=USE_DROGON=0 --cxxopt=-DUSE_DROGON=0 --cxxopt=-std=c++17 --host_cxxopt=-std=c++17 --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=1 --cxxopt=-DOVMS_DUMP_TO_FILE=0 --copt=-DGRPC_BAZEL_BUILD )

while [[ -n $1 ]]; do
  case $1 in
    -d)
      shift
      out_dir=$1
      ;;
    -b)
      build_only=true
      ;;
    -r)
      run_only=true
      ;;
    -t)
      one_target=true
      shift
      one_target_name=$1
      echo "one_target_name: $one_target_name"
      ;;
    *)
      echo "Unsupported input argument $1."
      exit 1
      ;;
  esac
  shift
done

echo "app_dir: $app_dir"
echo "out_dir: $out_dir"

declare -a bazel_flags

apps="${app_dir}/*"
for app in ${apps}; do
  if [[ -d "${app}" ]]; then
    target_name=${app##*/}
    if [[ "${target_name}" == "hello_world" ||
          "${target_name}" == "hello_ovms" ]]; then
      target="${app}:${target_name}"
    elif [[ "${target_name}" == "media_sequence" ]]; then
        target="${app}:${target_name}_demo"
        echo "Skipping target ${target}"
        continue
    elif [[ "${target_name}" == "autoflip" ]]; then
        target="${app}:run_${target_name}"
    elif [[ "${target_name}" == "object_detection_3d" ]]; then
        target="${app}:objectron_cpu"
    elif [[ "${target_name}" == "object_detection" ]]; then
        target="${app}:${target_name}_ovms"
    elif [[ "${target_name}" == "template_matching" ]]; then
        target="${app}:template_matching_tflite"
    elif [[ "${target_name}" == "youtube8m" ]]; then
        target="${app}:extract_yt8m_features"
        echo "Skipping target ${target}"
        continue
    else
      target="${app}:${target_name}_cpu"
    fi

    echo "target_name:${target_name}"
    echo "target:${target}"
    if [[ $one_target == true ]]; then
      if [[ "${target_name}" == "${one_target_name}" ]]; then
        bazel_flags=("${default_bazel_flags[@]}")
        bazel_flags+=(${target})
        echo "BUILD COMMAND:bazelisk ${bazel_flags[@]}"
        bazelisk "${bazel_flags[@]}"
        exit 0
      else
        continue
      fi
    fi

    if [[ $run_only == false ]]; then
      bazel_flags=("${default_bazel_flags[@]}")
      bazel_flags+=(${target})

      bazelisk "${bazel_flags[@]}"
    fi

    if [[ $build_only == false ]]; then
      cp -f "${bin_dir}/${app}/"*"" "${out_dir}"
      if  [[ ${target_name} == "object_tracking" ]]; then
        graph_name="tracking/object_detection_tracking"
      elif [[ ${target_name} == "upper_body_pose_tracking" ]]; then
        graph_name="pose_tracking/upper_body_pose_tracking"
      else
        graph_name="${target_name}/${target_name}"
      fi
      if [[ ${target_name} == "holistic_tracking" ||
            ${target_name} == "iris_tracking" ||
            ${target_name} == "pose_tracking" ||
            ${target_name} == "selfie_segmentation" ||
            ${target_name} == "upper_body_pose_tracking" ]]; then
        graph_suffix="cpu"
      else
        graph_suffix="desktop_live"
      fi
      GLOG_logtostderr=1 "${out_dir}/${target_name}_cpu" \
        --calculator_graph_config_file=mediapipe/graphs/"${graph_name}_${graph_suffix}.pbtxt"
    fi
  fi
done
