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
# Script to build all MediaPipe iOS example apps.
#
# To build all apps and store them in out_dir:
#   $ ./build_ios_examples.sh -d out_dir
#   Omitting -d and the associated directory saves all generated IPAs in the
#   current directory.
#   $ ./build_ios_examples.sh -d out_dir --nostrip
#   Same as above except that the symnbols are not stripped.

set -e

out_dir="."
strip=true
app_dir="mediapipe/examples/ios"
bin_dir="bazel-bin"
declare -a default_bazel_flags=(build -c opt --config=ios_arm64)

while [[ -n $1 ]]; do
  case $1 in
    -d)
      shift
      out_dir=$1
      ;;
    --nostrip)
      strip=false
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

declare -a bazel_flags

apps="${app_dir}/*"
for app in ${apps}; do
  if [[ -d "${app}" ]]; then
    target_name=${app##*/}
    if [[ "${target_name}" == "common" ]]; then
      continue
    fi
    target="${app}:${target_name}"

    echo "=== Target: ${target}"

    bazel_flags=("${default_bazel_flags[@]}")
    bazel_flags+=(${target})
    if [[ $strip == true ]]; then
      bazel_flags+=(--linkopt=-s)
    fi

    bazelisk "${bazel_flags[@]}"
    cp -f "${bin_dir}/${app}/"*".ipa" "${out_dir}"
  fi
done
