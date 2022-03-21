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
# Script to build all MediaPipe Android example apps.
#
# To build all apps and store them in out_dir, and install them:
#   $ ./build_android_examples.sh -d out_dir
#   Omitting -d and the associated directory saves all generated APKs in the
#   current directory.
#   $ ./build_android_examples.sh -d out_dir --nostrip
#   Same as above except that the symnbols are not stripped.
#
# To install the apps already stored in out_dir (after building them with the
# usages above):
#   $ ./build_android_examples.sh -d out_dir -i
#   Omitting -d and the associated directory assumes the apps are in the
#   current directory.

set -e

function switch_to_opencv_3() {
  echo "Switching to OpenCV 3"
  sed -i -e 's:4.0.1/opencv-4.0.1:3.4.3/opencv-3.4.3:g' WORKSPACE
  sed -i -e 's:libopencv_java4:libopencv_java3:g' third_party/opencv_android.BUILD
}

function switch_to_opencv_4() {
  echo "Switching to OpenCV 4"
  sed -i -e 's:3.4.3/opencv-3.4.3:4.0.1/opencv-4.0.1:g' WORKSPACE
  sed -i -e 's:libopencv_java3:libopencv_java4:g' third_party/opencv_android.BUILD
}

out_dir="."
strip=true
install_only=false
app_dir="mediapipe/examples/android/src/java/com/google/mediapipe/apps"
bin_dir="bazel-bin"
declare -a default_bazel_flags=(build -c opt --config=android_arm64)

while [[ -n $1 ]]; do
  case $1 in
    -d)
      shift
      out_dir=$1
      ;;
    --nostrip)
      strip=false
      ;;
    -i)
      install_only=true
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
echo "strip: $strip"

declare -a apks=()
declare -a bazel_flags
switch_to_opencv_3

apps="${app_dir}/*"
for app in ${apps}; do
  if [[ -d "${app}" ]]; then
    app_name=${app##*/}
    if [[ ${app_name} == "basic" ]]; then
      target_name="helloworld"
    else
      target_name=${app_name}
    fi
    target="${app}:${target_name}"
    bin="${bin_dir}/${app}/${target_name}.apk"

    echo "=== Target: ${target}"

    if [[ $install_only == false ]]; then
      bazel_flags=("${default_bazel_flags[@]}")
      bazel_flags+=(${target})
      if [[ $strip == true ]]; then
        bazel_flags+=(--linkopt=-s)
      fi
    fi

    if [[ ${app_name} == "objectdetection3d" ]]; then
      categories=("shoe" "chair" "cup" "camera" "shoe_1stage" "chair_1stage")
      for category in "${categories[@]}"; do
        apk="${out_dir}/${target_name}_${category}.apk"
        if [[ $install_only == false ]]; then
          bazel_flags_extended=("${bazel_flags[@]}")
          if [[ ${category} != "shoe" ]]; then
            bazel_flags_extended+=(--define ${category}=true)
          fi
          bazelisk "${bazel_flags_extended[@]}"
          cp -f "${bin}" "${apk}"
        fi
        apks+=(${apk})
      done
    else
      apk="${out_dir}/${target_name}.apk"
      if [[ $install_only == false ]]; then
        if [[ ${app_name} == "templatematchingcpu" ]]; then
          switch_to_opencv_4
        fi
        bazelisk "${bazel_flags[@]}"
        cp -f "${bin}" "${apk}"
        if [[ ${app_name} == "templatematchingcpu" ]]; then
          switch_to_opencv_3
        fi
      fi
      apks+=(${apk})
    fi
  fi
done

echo
echo "Connect your device via adb to install the apps."
read -p "Press 'a' to abort, or press any other key to continue ..." -n 1 -r
echo
if [[ ! $REPLY =~ ^[Aa]$ ]]; then
  for apk in "${apks[@]}"; do
    echo "=== Installing $apk"
    adb install -r "${apk}"
  done
fi
