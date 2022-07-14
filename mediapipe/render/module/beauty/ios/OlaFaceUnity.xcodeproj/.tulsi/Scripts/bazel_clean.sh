#!/bin/bash
# Copyright 2016 The Tulsi Authors. All rights reserved.
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
#
# Bridge between Xcode and Bazel for the "clean" action.
#
# Usage: bazel_clean.sh <bazel_binary_path> <bazel_binary_output_path> <bazel startup options>
# Note that the ACTION environment variable is expected to be set to "clean".

set -eu

readonly bazel_executable="$1"; shift
readonly bazel_bin_dir="$1"; shift

if [ -z $# ]; then
  readonly arguments=(clean)
else
  readonly arguments=("$@" clean)
fi

if [[ "${ACTION}" != "clean" ]]; then
  exit 0
fi

# Removes a directory if it exists and is not a symlink.
function remove_dir() {
  directory="$1"

  if [[ -d "${directory}" && ! -L "${directory}" ]]; then
    rm -r "${directory}"
  fi
}

# Xcode may have generated a bazel-bin directory after a previous clean.
# Remove it to prevent a useless warning.
remove_dir "${bazel_bin_dir}"

(
  set -x
  "${bazel_executable}" "${arguments[@]}"
)

